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
from itertools import chain
from typing import Annotated, Optional, Sequence, TypeAlias
from fastapi import APIRouter, HTTPException, Path, status, Query
from pydantic import Field

from parlant.api import agents, common
from parlant.api.common import (
    InvoiceDataDTO,
    PayloadKindDTO,
    ToolIdDTO,
    apigen_config,
    apigen_skip_config,
)
from parlant.api.index import InvoiceDTO
from parlant.core.agents import AgentStore
from parlant.core.common import (
    DefaultBaseModel,
)
from parlant.api.common import (
    ExampleJson,
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
from parlant.core.guidelines import (
    Guideline,
    GuidelineContent,
    GuidelineId,
    GuidelineStore,
    GuidelineUpdateParams,
)
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationId,
    GuidelineToolAssociationStore,
)
from parlant.core.application import Application
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import TagId, TagStore, Tag
from parlant.core.tools import ToolId

from parlant.api.common import (
    GuidelineConditionField,
    GuidelineActionField,
)

API_GROUP = "guidelines"


GuidelineIdPath: TypeAlias = Annotated[
    GuidelineId,
    Path(
        description="Unique identifier for the guideline",
        examples=["IUCGT-l4pS"],
    ),
]


GuidelineEnabledField: TypeAlias = Annotated[
    bool,
    Field(
        default=True,
        description="Whether the guideline is enabled",
        examples=[True, False],
    ),
]

legacy_guideline_dto_example: ExampleJson = {
    "id": "guid_123xz",
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
    "enabled": True,
}


class LegacyGuidelineDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_guideline_dto_example},
):
    """Assigns an id to the condition-action pair"""

    id: GuidelineIdPath
    condition: GuidelineConditionField
    action: GuidelineActionField
    enabled: GuidelineEnabledField


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


legacy_guideline_connection_dto_example: ExampleJson = {
    "id": "conn_456xyz",
    "source": {
        "id": "guid_123xz",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
        "enabled": True,
    },
    "target": {
        "id": "guid_789yz",
        "condition": "when providing pricing information",
        "action": "mention any seasonal discounts",
        "enabled": True,
    },
    "indirect": False,
}


class LegacyGuidelineConnectionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_guideline_connection_dto_example},
):
    """
    Represents a connection between two guidelines.

    """

    id: GuidelineConnectionIdField
    source: LegacyGuidelineDTO
    target: LegacyGuidelineDTO
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


legacy_guideline_with_connections_example: ExampleJson = {
    "guideline": {
        "id": "guid_123xz",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
        "enabled": True,
    },
    "connections": [
        {
            "id": "conn_456yz",
            "source": {
                "id": "guid_123xz",
                "condition": "when the customer asks about pricing",
                "action": "provide current pricing information",
                "enabled": True,
            },
            "target": {
                "id": "guid_789yz",
                "condition": "when providing pricing information",
                "action": "mention any seasonal discounts",
                "enabled": True,
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


class LegacyGuidelineWithConnectionsAndToolAssociationsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_guideline_with_connections_example},
):
    """A Guideline with its connections and tool associations."""

    guideline: LegacyGuidelineDTO
    connections: Sequence[LegacyGuidelineConnectionDTO]
    tool_associations: Sequence[GuidelineToolAssociationDTO]


legacy_guideline_creation_params_example: ExampleJson = {
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


class LegacyGuidelineCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_guideline_creation_params_example},
):
    """Evaluation invoices to generate Guidelines from."""

    invoices: Sequence[InvoiceDTO]


legacy_guideline_creation_result_example: ExampleJson = {
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


class LegacyGuidelineCreationResult(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_guideline_creation_result_example},
):
    """Result wrapper for Guidelines creation."""

    items: Sequence[LegacyGuidelineWithConnectionsAndToolAssociationsDTO]


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
    "enabled": True,
}


class LegacyGuidelineUpdateParamsDTO(
    DefaultBaseModel, json_schema_extra={"example": guideline_update_params_example}
):
    """Parameters for updating Guideline objects."""

    connections: Optional[GuidelineConnectionUpdateParamsDTO] = None
    tool_associations: Optional[GuidelineToolAssociationUpdateParamsDTO] = None
    enabled: Optional[bool] = None


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


async def _get_guideline_connections(
    guideline_store: GuidelineStore,
    guideline_connection_store: GuidelineConnectionStore,
    guideline_id: GuidelineId,
    include_indirect: bool = True,
) -> Sequence[tuple[_GuidelineConnection, bool]]:
    connections = [
        _GuidelineConnection(
            id=c.id,
            source=await guideline_store.read_guideline(guideline_id=c.source),
            target=await guideline_store.read_guideline(guideline_id=c.target),
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


def create_legacy_router(
    application: Application,
    guideline_store: GuidelineStore,
    guideline_connection_store: GuidelineConnectionStore,
    service_registry: ServiceRegistry,
    guideline_tool_association_store: GuidelineToolAssociationStore,
) -> APIRouter:
    """
    DEPRECATED: This router uses agent-based paths which are being phased out.
    Use the tag-based API instead.
    """
    router = APIRouter()

    @router.post(
        "/{agent_id}/guidelines",
        status_code=status.HTTP_201_CREATED,
        operation_id="legacy_create_guidelines",
        response_model=LegacyGuidelineCreationResult,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Guidelines successfully created. Returns the created guidelines with their connections and tool associations.",
                "content": common.example_json_content(legacy_guideline_creation_result_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        deprecated=True,
        **apigen_skip_config(),
    )
    async def create_guidelines(
        agent_id: agents.AgentIdPath,
        params: LegacyGuidelineCreationParamsDTO,
    ) -> LegacyGuidelineCreationResult:
        """
        DEPRECATED: Use the tag-based API instead.

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
                invoices=invoices,
            )
        )

        for id in guideline_ids:
            _ = await guideline_store.upsert_tag(
                guideline_id=id,
                tag_id=Tag.for_agent_id(agent_id),
            )

        guidelines = [await guideline_store.read_guideline(guideline_id=id) for id in guideline_ids]

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

        return LegacyGuidelineCreationResult(
            items=[
                LegacyGuidelineWithConnectionsAndToolAssociationsDTO(
                    guideline=LegacyGuidelineDTO(
                        id=guideline.id,
                        condition=guideline.content.condition,
                        action=guideline.content.action,
                        enabled=guideline.enabled,
                    ),
                    connections=[
                        LegacyGuidelineConnectionDTO(
                            id=connection.id,
                            source=LegacyGuidelineDTO(
                                id=connection.source.id,
                                condition=connection.source.content.condition,
                                action=connection.source.content.action,
                                enabled=connection.source.enabled,
                            ),
                            target=LegacyGuidelineDTO(
                                id=connection.target.id,
                                condition=connection.target.content.condition,
                                action=connection.target.content.action,
                                enabled=connection.target.enabled,
                            ),
                            indirect=indirect,
                        )
                        for connection, indirect in await _get_guideline_connections(
                            guideline_store=guideline_store,
                            guideline_connection_store=guideline_connection_store,
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
        operation_id="legacy_read_guideline",
        response_model=LegacyGuidelineWithConnectionsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline details successfully retrieved. Returns the complete guideline with its connections and tool associations.",
                "content": common.example_json_content(legacy_guideline_with_connections_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline or agent not found"},
        },
        deprecated=True,
        **apigen_skip_config(),
    )
    async def read_guideline(
        agent_id: agents.AgentIdPath,
        guideline_id: GuidelineIdPath,
    ) -> LegacyGuidelineWithConnectionsAndToolAssociationsDTO:
        """
        DEPRECATED: Use the tag-based API instead.

        Retrieves a specific guideline with all its connections and tool associations.

        Returns both direct and indirect connections between guidelines.
        Tool associations indicate which tools the guideline can use.
        """
        guidelines = await guideline_store.list_guidelines(
            tags=[Tag.for_agent_id(agent_id)],
        )

        guideline = next(
            (g for g in guidelines if g.id == guideline_id),
            None,
        )

        if not guideline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline is not associated with the specified agent",
            )

        connections = await _get_guideline_connections(
            guideline_store=guideline_store,
            guideline_connection_store=guideline_connection_store,
            guideline_id=guideline_id,
            include_indirect=True,
        )

        return LegacyGuidelineWithConnectionsAndToolAssociationsDTO(
            guideline=LegacyGuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                enabled=guideline.enabled,
            ),
            connections=[
                LegacyGuidelineConnectionDTO(
                    id=connection.id,
                    source=LegacyGuidelineDTO(
                        id=connection.source.id,
                        condition=connection.source.content.condition,
                        action=connection.source.content.action,
                        enabled=connection.source.enabled,
                    ),
                    target=LegacyGuidelineDTO(
                        id=connection.target.id,
                        condition=connection.target.content.condition,
                        action=connection.target.content.action,
                        enabled=connection.target.enabled,
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
        operation_id="legacy_list_guidelines",
        response_model=Sequence[LegacyGuidelineDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all guidelines for the specified agent",
                "content": common.example_json_content([legacy_guideline_dto_example]),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
        },
        deprecated=True,
        **apigen_skip_config(),
    )
    async def list_guidelines(
        agent_id: agents.AgentIdPath,
    ) -> Sequence[LegacyGuidelineDTO]:
        """
        DEPRECATED: Use the tag-based API instead.

        Lists all guidelines for the specified agent.

        Returns an empty list if no guidelines exist.
        Guidelines are returned in no guaranteed order.
        Does not include connections or tool associations.
        """
        guidelines = await guideline_store.list_guidelines(
            tags=[Tag.for_agent_id(agent_id)],
        )

        return [
            LegacyGuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                enabled=guideline.enabled,
            )
            for guideline in guidelines
        ]

    @router.patch(
        "/{agent_id}/guidelines/{guideline_id}",
        operation_id="legacy_update_guideline",
        response_model=LegacyGuidelineWithConnectionsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline successfully updated. Returns the updated guideline with its connections and tool associations.",
                "content": common.example_json_content(legacy_guideline_with_connections_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Guideline, agent, or referenced tool not found"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Invalid connection rules or validation error in update parameters"
            },
        },
        deprecated=True,
        **apigen_skip_config(),
    )
    async def update_guideline(
        agent_id: agents.AgentIdPath,
        guideline_id: GuidelineIdPath,
        params: LegacyGuidelineUpdateParamsDTO,
    ) -> LegacyGuidelineWithConnectionsAndToolAssociationsDTO:
        """
        DEPRECATED: Use the tag-based API instead.

        Updates a guideline's connections and tool associations.

        Only provided attributes will be updated; others remain unchanged.

        Connection rules:
        - A guideline cannot connect to itself
        - Only direct connections can be removed
        - The connection must specify this guideline as source or target

        Tool Association rules:
        - Tool services and tools must exist before creating associations
        """
        guidelines = await guideline_store.list_guidelines(
            tags=[Tag.for_agent_id(agent_id)],
        )
        guideline = next(
            (g for g in guidelines if g.id == guideline_id),
            None,
        )

        if not guideline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Updated guideline not found for the specified agent",
            )

        if params.enabled is not None:
            await guideline_store.update_guideline(
                guideline_id=guideline_id,
                params=GuidelineUpdateParams(enabled=params.enabled),
            )

        guidelines = await guideline_store.list_guidelines(
            tags=[Tag.for_agent_id(agent_id)],
        )

        if params.connections and params.connections.add:
            for req in params.connections.add:
                if req.source == req.target:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="A guideline cannot be connected to itself",
                    )
                elif req.source == guideline.id:
                    if not any(g.id == req.target for g in guidelines):
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail="The target guideline is not associated with the specified agent",
                        )
                elif req.target == guideline.id:
                    if not any(g.id == req.source for g in guidelines):
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail="The source guideline is not associated with the specified agent",
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

        connections = await _get_guideline_connections(
            guideline_store=guideline_store,
            guideline_connection_store=guideline_connection_store,
            guideline_id=guideline_id,
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

        updated_guideline = await guideline_store.read_guideline(guideline_id=guideline_id)

        return LegacyGuidelineWithConnectionsAndToolAssociationsDTO(
            guideline=LegacyGuidelineDTO(
                id=updated_guideline.id,
                condition=updated_guideline.content.condition,
                action=updated_guideline.content.action,
                enabled=updated_guideline.enabled,
            ),
            connections=[
                LegacyGuidelineConnectionDTO(
                    id=connection.id,
                    source=LegacyGuidelineDTO(
                        id=connection.source.id,
                        condition=connection.source.content.condition,
                        action=connection.source.content.action,
                        enabled=connection.source.enabled,
                    ),
                    target=LegacyGuidelineDTO(
                        id=connection.target.id,
                        condition=connection.target.content.condition,
                        action=connection.target.content.action,
                        enabled=connection.target.enabled,
                    ),
                    indirect=indirect,
                )
                for connection, indirect in await _get_guideline_connections(
                    guideline_store=guideline_store,
                    guideline_connection_store=guideline_connection_store,
                    guideline_id=guideline_id,
                    include_indirect=True,
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
        operation_id="legacy_delete_guideline",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Guideline successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline or agent not found"},
        },
        deprecated=True,
        **apigen_skip_config(),
    )
    async def delete_guideline(
        agent_id: agents.AgentIdPath,
        guideline_id: GuidelineIdPath,
    ) -> None:
        """
        DEPRECATED: Use the tag-based API instead.

        Deletes a guideline from the agent.

        Also removes all associated connections and tool associations.
        Deleting a non-existent guideline will return 404.
        No content will be returned from a successful deletion.
        """
        guidelines = await guideline_store.list_guidelines(
            tags=[Tag.for_agent_id(agent_id)],
        )

        if not any(g.id == guideline_id for g in guidelines):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline is not associated with the specified agent",
            )

        await guideline_store.remove_tag(
            guideline_id=guideline_id,
            tag_id=Tag.for_agent_id(agent_id),
        )

        updated_guideline = await guideline_store.read_guideline(guideline_id=guideline_id)

        deleted = False
        if not updated_guideline.tags:
            await guideline_store.delete_guideline(guideline_id=guideline_id)
            deleted = True
        for c in chain(
            await guideline_connection_store.list_connections(indirect=False, source=guideline_id),
            await guideline_connection_store.list_connections(indirect=False, target=guideline_id),
        ):
            if deleted:
                await guideline_connection_store.delete_connection(c.id)
            else:
                connected_guideline = (
                    await guideline_store.read_guideline(c.target)
                    if c.source == guideline_id
                    else await guideline_store.read_guideline(c.source)
                )
                if connected_guideline.tags and not any(
                    t in connected_guideline.tags for t in updated_guideline.tags
                ):
                    await guideline_connection_store.delete_connection(c.id)

        for associastion in await guideline_tool_association_store.list_associations():
            if associastion.guideline_id == guideline_id:
                await guideline_tool_association_store.delete_association(associastion.id)

    return router


TagIdQuery: TypeAlias = Annotated[
    Optional[TagId],
    Query(
        description="The tag ID to filter guidelines by",
        examples=["tag:123"],
    ),
]

GuidelineTagsField: TypeAlias = Annotated[
    Sequence[TagId],
    Field(
        description="The tags associated with the guideline",
        examples=[["tag1", "tag2"], []],
    ),
]

guideline_creation_params_example: ExampleJson = {
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
    "enabled": False,
}


class GuidelineCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_creation_params_example},
):
    """Parameters for creating a new guideline."""

    condition: GuidelineConditionField
    action: GuidelineActionField
    enabled: Optional[GuidelineEnabledField] = None
    tags: Optional[GuidelineTagsField] = None


GuidelineTagsUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to add to the guideline",
        examples=[["tag1", "tag2"]],
    ),
]

GuidelineTagsUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to remove from the guideline",
        examples=[["tag1", "tag2"]],
    ),
]

guideline_tags_update_params_example: ExampleJson = {
    "add": [
        "tag1",
        "tag2",
    ],
    "remove": [
        "tag3",
        "tag4",
    ],
}


class GuidelineTagsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_tags_update_params_example},
):
    """
    Parameters for updating the tags of an existing guideline.
    """

    add: Optional[GuidelineTagsUpdateAddField] = None
    remove: Optional[GuidelineTagsUpdateRemoveField] = None


guideline_dto_example = {
    "id": "guid_123xz",
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
    "enabled": True,
    "tags": ["tag1", "tag2"],
}


class GuidelineDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_dto_example},
):
    """Represents a guideline."""

    id: GuidelineIdPath
    condition: GuidelineConditionField
    action: GuidelineActionField
    enabled: GuidelineEnabledField
    tags: GuidelineTagsField


class GuidelineUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_update_params_example},
):
    """Parameters for updating a guideline."""

    condition: Optional[GuidelineConditionField] = None
    action: Optional[GuidelineActionField] = None
    connections: Optional[GuidelineConnectionUpdateParamsDTO] = None
    tool_associations: Optional[GuidelineToolAssociationUpdateParamsDTO] = None
    enabled: Optional[GuidelineEnabledField] = None
    tags: Optional[GuidelineTagsUpdateParamsDTO] = None


guideline_connection_example: ExampleJson = {
    "id": "123",
    "source": "456",
    "target": "789",
    "indirect": False,
}


class GuidelineConnectionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_connection_example},
):
    """Represents a connection between two guidelines."""

    id: GuidelineConnectionIdField
    source: GuidelineDTO
    target: GuidelineDTO
    indirect: GuidelineConnectionIndirectField


guideline_with_connections_example: ExampleJson = {
    "guideline": {
        "id": "guid_123xz",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
        "enabled": True,
        "tags": ["tag1", "tag2"],
    },
    "connections": [
        {
            "id": "conn_456yz",
            "source": {
                "id": "guid_123xz",
                "condition": "when the customer asks about pricing",
                "action": "provide current pricing information",
                "enabled": True,
                "tags": ["tag1", "tag2"],
            },
            "target": {
                "id": "guid_789yz",
                "condition": "when providing pricing information",
                "action": "mention any seasonal discounts",
                "enabled": True,
                "tags": ["tag1", "tag2"],
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


def create_router(
    guideline_store: GuidelineStore,
    guideline_connection_store: GuidelineConnectionStore,
    service_registry: ServiceRegistry,
    guideline_tool_association_store: GuidelineToolAssociationStore,
    agent_store: AgentStore,
    tag_store: TagStore,
) -> APIRouter:
    """Creates a router for the guidelines API with tag-based paths."""
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_guideline",
        response_model=GuidelineDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Guideline successfully created. Returns the created guideline.",
                "content": common.example_json_content(guideline_dto_example),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_guideline(
        params: GuidelineCreationParamsDTO,
    ) -> GuidelineDTO:
        """
        Creates a new guideline.

        See the [documentation](https://parlant.io/docs/concepts/customization/guidelines) for more information.
        """
        tags = []
        if params.tags:
            for tag_id in params.tags:
                if agent_id := Tag.extract_agent_id(tag_id):
                    _ = await agent_store.read_agent(agent_id=agent_id)
                else:
                    _ = await tag_store.read_tag(tag_id=tag_id)

            tags = list(set(params.tags))

        guideline = await guideline_store.create_guideline(
            condition=params.condition,
            action=params.action,
            enabled=params.enabled or True,
            tags=tags or None,
        )

        return GuidelineDTO(
            id=guideline.id,
            condition=guideline.content.condition,
            action=guideline.content.action,
            enabled=guideline.enabled,
            tags=guideline.tags,
        )

    @router.get(
        "",
        operation_id="list_guidelines",
        response_model=Sequence[GuidelineDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all guidelines for the specified tag or all guidelines if no tag is provided",
                "content": common.example_json_content([guideline_dto_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_guidelines(
        tag_id: TagIdQuery = None,
    ) -> Sequence[GuidelineDTO]:
        """
        Lists all guidelines for the specified tag or all guidelines if no tag is provided.

        Returns an empty list if no guidelines exist.
        Guidelines are returned in no guaranteed order.
        Does not include connections or tool associations.
        """
        if tag_id:
            guidelines = await guideline_store.list_guidelines(
                tags=[tag_id],
            )
        else:
            guidelines = await guideline_store.list_guidelines()

        return [
            GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                enabled=guideline.enabled,
                tags=guideline.tags,
            )
            for guideline in guidelines
        ]

    @router.get(
        "/{guideline_id}",
        operation_id="read_guideline",
        response_model=GuidelineWithConnectionsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline details successfully retrieved. Returns the complete guideline with its connections and tool associations.",
                "content": common.example_json_content(guideline_with_connections_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_guideline(
        guideline_id: GuidelineIdPath,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
        """
        Retrieves a specific guideline with all its connections and tool associations.

        Returns both direct and indirect connections between guidelines.
        Tool associations indicate which tools the guideline can use.
        """
        try:
            guideline = await guideline_store.read_guideline(guideline_id=guideline_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline not found",
            )

        connections = await _get_guideline_connections(
            guideline_store=guideline_store,
            guideline_connection_store=guideline_connection_store,
            guideline_id=guideline_id,
            include_indirect=True,
        )

        return GuidelineWithConnectionsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=guideline.id,
                condition=guideline.content.condition,
                action=guideline.content.action,
                enabled=guideline.enabled,
                tags=guideline.tags,
            ),
            connections=[
                GuidelineConnectionDTO(
                    id=connection.id,
                    source=GuidelineDTO(
                        id=connection.source.id,
                        condition=connection.source.content.condition,
                        action=connection.source.content.action,
                        enabled=connection.source.enabled,
                        tags=connection.source.tags,
                    ),
                    target=GuidelineDTO(
                        id=connection.target.id,
                        condition=connection.target.content.condition,
                        action=connection.target.content.action,
                        enabled=connection.target.enabled,
                        tags=connection.target.tags,
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

    @router.patch(
        "/{guideline_id}",
        operation_id="update_guideline",
        response_model=GuidelineWithConnectionsAndToolAssociationsDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Guideline successfully updated. Returns the updated guideline with its connections and tool associations.",
                "content": common.example_json_content(guideline_with_connections_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline or referenced tool not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Invalid connection rules or validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_guideline(
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
        _ = await guideline_store.read_guideline(guideline_id=guideline_id)

        if params.condition or params.action or params.enabled is not None:
            update_params: GuidelineUpdateParams = {}
            if params.condition:
                update_params["condition"] = params.condition
            if params.action:
                update_params["action"] = params.action
            if params.enabled is not None:
                update_params["enabled"] = params.enabled

            await guideline_store.update_guideline(
                guideline_id=guideline_id,
                params=GuidelineUpdateParams(**update_params),
            )

        if params.connections and params.connections.add:
            for req in params.connections.add:
                if req.source == req.target:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="A guideline cannot be connected to itself",
                    )
                elif req.source != guideline_id and req.target != guideline_id:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="The connection must specify the guideline at hand as either source or target",
                    )

                await guideline_connection_store.create_connection(
                    source=req.source,
                    target=req.target,
                )

        connections = await _get_guideline_connections(
            guideline_store=guideline_store,
            guideline_connection_store=guideline_connection_store,
            guideline_id=guideline_id,
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

                try:
                    service = await service_registry.read_tool_service(service_name)
                    _ = await service.read_tool(tool_name)
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tool not found for service '{service_name}' and tool '{tool_name}'",
                    )

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
                        and assoc.guideline_id == guideline_id
                    ),
                    None,
                ):
                    await guideline_tool_association_store.delete_association(association.id)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tool association not found for service '{tool_id_dto.service_name}' and tool '{tool_id_dto.tool_name}'",
                    )

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    if agent_id := Tag.extract_agent_id(tag_id):
                        _ = await agent_store.read_agent(agent_id=agent_id)
                    else:
                        _ = await tag_store.read_tag(tag_id=tag_id)
                        await guideline_store.upsert_tag(
                            guideline_id=guideline_id,
                            tag_id=tag_id,
                        )
            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await guideline_store.remove_tag(
                        guideline_id=guideline_id,
                        tag_id=tag_id,
                    )

        updated_guideline = await guideline_store.read_guideline(guideline_id=guideline_id)

        return GuidelineWithConnectionsAndToolAssociationsDTO(
            guideline=GuidelineDTO(
                id=updated_guideline.id,
                condition=updated_guideline.content.condition,
                action=updated_guideline.content.action,
                enabled=updated_guideline.enabled,
                tags=updated_guideline.tags,
            ),
            connections=[
                GuidelineConnectionDTO(
                    id=connection.id,
                    source=GuidelineDTO(
                        id=connection.source.id,
                        condition=connection.source.content.condition,
                        action=connection.source.content.action,
                        enabled=connection.source.enabled,
                        tags=connection.source.tags,
                    ),
                    target=GuidelineDTO(
                        id=connection.target.id,
                        condition=connection.target.content.condition,
                        action=connection.target.content.action,
                        enabled=connection.target.enabled,
                        tags=connection.target.tags,
                    ),
                    indirect=indirect,
                )
                for connection, indirect in await _get_guideline_connections(
                    guideline_store=guideline_store,
                    guideline_connection_store=guideline_connection_store,
                    guideline_id=guideline_id,
                    include_indirect=True,
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
        "/{guideline_id}",
        operation_id="delete_guideline",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Guideline successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {"description": "Guideline not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_guideline(
        guideline_id: GuidelineIdPath,
    ) -> None:
        guideline = await guideline_store.read_guideline(guideline_id=guideline_id)

        await guideline_store.delete_guideline(guideline_id=guideline_id)

        for c in chain(
            await guideline_connection_store.list_connections(indirect=False, source=guideline_id),
            await guideline_connection_store.list_connections(indirect=False, target=guideline_id),
        ):
            connected_guideline = (
                await guideline_store.read_guideline(c.target)
                if c.source == guideline_id
                else await guideline_store.read_guideline(c.source)
            )
            if connected_guideline.tags and not any(
                t in connected_guideline.tags for t in guideline.tags
            ):
                await guideline_connection_store.delete_connection(c.id)

        for associastion in await guideline_tool_association_store.list_associations():
            if associastion.guideline_id == guideline_id:
                await guideline_tool_association_store.delete_association(associastion.id)

    return router
