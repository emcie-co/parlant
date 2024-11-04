from collections import defaultdict
from itertools import chain
from typing import Optional, Sequence
from fastapi import APIRouter, HTTPException, status


from parlant.api.common import (
    ConnectionKindDTO,
    GuidelinePayloadDTO,
    GuidelineInvoiceDataDTO,
    ToolIdDTO,
    connection_kind_dto_to_connection_kind,
    connection_kind_to_dto,
)
from parlant.core.agents import AgentId
from parlant.core.common import DefaultBaseModel
from parlant.core.evaluations import (
    CoherenceCheck,
    ConnectionProposition,
    GuidelinePayload,
    Invoice,
    InvoiceGuidelineData,
    PayloadKind,
)
from parlant.core.guideline_connections import (
    ConnectionKind,
    GuidelineConnectionId,
    GuidelineConnectionStore,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationId,
    GuidelineToolAssociationStore,
)
from parlant.core.application import Application
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import ToolId


class GuidelineDTO(DefaultBaseModel):
    id: GuidelineId
    predicate: str
    action: str


class GuidelineConnectionDTO(DefaultBaseModel):
    id: GuidelineConnectionId
    source: GuidelineDTO
    target: GuidelineDTO
    kind: ConnectionKindDTO
    indirect: bool


class GuidelineToolAssociationDTO(DefaultBaseModel):
    id: GuidelineToolAssociationId
    guideline_id: GuidelineId
    tool_id: ToolIdDTO


class GuidelineWithConnectionsAndToolAssociationsDTO(DefaultBaseModel):
    guideline: GuidelineDTO
    connections: Sequence[GuidelineConnectionDTO]
    tool_associations: Sequence[GuidelineToolAssociationDTO]


class GuidelineInvoiceDTO(DefaultBaseModel):
    payload: GuidelinePayloadDTO
    checksum: str
    approved: bool
    data: GuidelineInvoiceDataDTO
    error: Optional[str]


class CreateGuidelineRequest(DefaultBaseModel):
    invoices: Sequence[GuidelineInvoiceDTO]


class CreateGuidelinesResponse(DefaultBaseModel):
    items: list[GuidelineWithConnectionsAndToolAssociationsDTO]


class DeleteGuidelineRequest(DefaultBaseModel):
    guideline_id: GuidelineId


class DeleteGuidelineResponse(DefaultBaseModel):
    guideline_id: GuidelineId


class ListGuidelinesResponse(DefaultBaseModel):
    guidelines: list[GuidelineDTO]


class GuidelineConnectionAdditionDTO(DefaultBaseModel):
    source: GuidelineId
    target: GuidelineId
    kind: ConnectionKindDTO


class GuidelineConnectionsPatchDTO(DefaultBaseModel):
    add: Optional[Sequence[GuidelineConnectionAdditionDTO]] = None
    remove: Optional[Sequence[GuidelineId]] = None


class GuidelineToolAssociationsPatchDTO(DefaultBaseModel):
    add: Optional[Sequence[ToolIdDTO]] = None
    remove: Optional[Sequence[ToolIdDTO]] = None


class PatchGuidelineRequest(DefaultBaseModel):
    connections: Optional[GuidelineConnectionsPatchDTO] = None
    tool_associations: Optional[GuidelineToolAssociationsPatchDTO] = None


class GuidelineConnection(DefaultBaseModel):
    id: GuidelineConnectionId
    source: Guideline
    target: Guideline
    kind: ConnectionKind


def _invoice_dto_to_invoice(dto: GuidelineInvoiceDTO) -> Invoice:
    if not dto.approved:
        raise ValueError("Unapproved invoice.")

    payload = GuidelinePayload(
        content=GuidelineContent(
            predicate=dto.payload.content["predicate"],
            action=dto.payload.content["action"],
        ),
        operation=dto.payload.operation,
        coherence_check=dto.payload.coherence_check,
        connection_proposition=dto.payload.connection_proposition,
        updated_id=dto.payload.updated_id,
    )

    kind = PayloadKind.GUIDELINE

    if not dto.data:
        raise ValueError("Unsupported payload.")

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


def _invoice_data_dto_to_invoice_data(dto: GuidelineInvoiceDataDTO) -> InvoiceGuidelineData:
    try:
        coherence_checks = [
            CoherenceCheck(
                kind=check.kind,
                first=GuidelineContent(predicate=check.first.predicate, action=check.first.action),
                second=GuidelineContent(
                    predicate=check.second.predicate, action=check.second.action
                ),
                issue=check.issue,
                severity=check.severity,
            )
            for check in dto.coherence_checks
        ]

        if dto.connection_propositions:
            connection_propositions = [
                ConnectionProposition(
                    check_kind=prop.check_kind,
                    source=GuidelineContent(
                        predicate=prop.source.predicate, action=prop.source.action
                    ),
                    target=GuidelineContent(
                        predicate=prop.target.predicate, action=prop.target.action
                    ),
                    connection_kind=connection_kind_dto_to_connection_kind(prop.connection_kind),
                )
                for prop in dto.connection_propositions
            ]
        else:
            connection_propositions = None

        return InvoiceGuidelineData(
            coherence_checks=coherence_checks, connection_propositions=connection_propositions
        )

    except Exception:
        raise ValueError(f"Unsupported invoice guideline data: {dto}")


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
    ) -> Sequence[tuple[GuidelineConnection, bool]]:
        connections = [
            GuidelineConnection(
                id=c.id,
                source=await guideline_store.read_guideline(
                    guideline_set=guideline_set, guideline_id=c.source
                ),
                target=await guideline_store.read_guideline(
                    guideline_set=guideline_set, guideline_id=c.target
                ),
                kind=c.kind,
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
    )
    async def create_guidelines(
        agent_id: AgentId,
        request: CreateGuidelineRequest,
    ) -> CreateGuidelinesResponse:
        try:
            invoices = [_invoice_dto_to_invoice(i) for i in request.invoices]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

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

        return CreateGuidelinesResponse(
            items=[
                GuidelineWithConnectionsAndToolAssociationsDTO(
                    guideline=GuidelineDTO(
                        id=guideline.id,
                        predicate=guideline.content.predicate,
                        action=guideline.content.action,
                    ),
                    connections=[
                        GuidelineConnectionDTO(
                            id=connection.id,
                            source=GuidelineDTO(
                                id=connection.source.id,
                                predicate=connection.source.content.predicate,
                                action=connection.source.content.action,
                            ),
                            target=GuidelineDTO(
                                id=connection.target.id,
                                predicate=connection.target.content.predicate,
                                action=connection.target.content.action,
                            ),
                            kind=connection_kind_to_dto(connection.kind),
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
    )
    async def read_guideline(
        agent_id: AgentId,
        guideline_id: GuidelineId,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
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
                predicate=guideline.content.predicate,
                action=guideline.content.action,
            ),
            connections=[
                GuidelineConnectionDTO(
                    id=connection.id,
                    source=GuidelineDTO(
                        id=connection.source.id,
                        predicate=connection.source.content.predicate,
                        action=connection.source.content.action,
                    ),
                    target=GuidelineDTO(
                        id=connection.target.id,
                        predicate=connection.target.content.predicate,
                        action=connection.target.content.action,
                    ),
                    kind=connection_kind_to_dto(connection.kind),
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
    )
    async def list_guidelines(agent_id: AgentId) -> ListGuidelinesResponse:
        guidelines = await guideline_store.list_guidelines(guideline_set=agent_id)

        return ListGuidelinesResponse(
            guidelines=[
                GuidelineDTO(
                    id=guideline.id,
                    predicate=guideline.content.predicate,
                    action=guideline.content.action,
                )
                for guideline in guidelines
            ]
        )

    @router.patch(
        "/{agent_id}/guidelines/{guideline_id}",
        operation_id="patch_guideline",
    )
    async def patch_guideline(
        agent_id: AgentId, guideline_id: GuidelineId, request: PatchGuidelineRequest
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
        guideline = await guideline_store.read_guideline(
            guideline_set=agent_id,
            guideline_id=guideline_id,
        )

        if request.connections and request.connections.add:
            for req in request.connections.add:
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
                    kind=connection_kind_dto_to_connection_kind(req.kind),
                )

        connections = await get_guideline_connections(
            agent_id,
            guideline_id,
            include_indirect=False,
        )

        if request.connections and request.connections.remove:
            for id in request.connections.remove:
                if found_connection := next(
                    (c for c, _ in connections if id in [c.source.id, c.target.id]), None
                ):
                    await guideline_connection_store.delete_connection(found_connection.id)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="Only direct connections may be removed",
                    )

        if request.tool_associations and request.tool_associations.add:
            for tool_id_dto in request.tool_associations.add:
                service_name = tool_id_dto.service_name
                tool_name = tool_id_dto.tool_name

                service = await service_registry.read_tool_service(service_name)
                _ = await service.read_tool(tool_name)

                await guideline_tool_association_store.create_association(
                    guideline_id=guideline_id,
                    tool_id=ToolId(service_name=service_name, tool_name=tool_name),
                )

        if request.tool_associations and request.tool_associations.remove:
            associations = await guideline_tool_association_store.list_associations()

            for tool_id_dto in request.tool_associations.remove:
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
                predicate=guideline.content.predicate,
                action=guideline.content.action,
            ),
            connections=[
                GuidelineConnectionDTO(
                    id=connection.id,
                    source=GuidelineDTO(
                        id=connection.source.id,
                        predicate=connection.source.content.predicate,
                        action=connection.source.content.action,
                    ),
                    target=GuidelineDTO(
                        id=connection.target.id,
                        predicate=connection.target.content.predicate,
                        action=connection.target.content.action,
                    ),
                    kind=connection_kind_to_dto(connection.kind),
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
        operation_id="delete_guideline",
    )
    async def delete_guideline(
        agent_id: AgentId,
        guideline_id: GuidelineId,
    ) -> DeleteGuidelineResponse:
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

        return DeleteGuidelineResponse(guideline_id=guideline_id)

    return router
