from datetime import datetime
from typing import Union
from fastapi import APIRouter
from typing_extensions import Literal

from emcie.server.core.common import DefaultBaseModel
from emcie.common.tools import Tool, ToolParameter, ToolId
from emcie.server.core.services.tools.openapi import OpenAPIClient
from emcie.server.core.services.tools.service_registry import ServiceRegistry, ToolServiceKind
from emcie.server.core.tools import ToolService


class CreateSDKServiceRequest(DefaultBaseModel):
    kind: Literal["sdk"]
    url: str


class CreateOpenAPIServiceRequest(DefaultBaseModel):
    kind: Literal["openapi"]
    url: str
    openapi_json: str


CreateServiceRequest = Union[CreateSDKServiceRequest, CreateOpenAPIServiceRequest]


class CreateSDKServiceResponse(DefaultBaseModel):
    name: str
    kind: Literal["sdk"] = "sdk"
    url: str


class CreateOpenAPIServiceResponse(DefaultBaseModel):
    name: str


class DeleteServiceResponse(DefaultBaseModel):
    name: str


class ServiceMetaDataDTO(DefaultBaseModel):
    name: str
    kind: ToolServiceKind
    url: str


class ListServicesResponse(DefaultBaseModel):
    services: list[ServiceMetaDataDTO]


class ToolDTO(DefaultBaseModel):
    id: ToolId
    creation_utc: datetime
    name: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str]
    consequential: bool


class ServiceDTO(DefaultBaseModel):
    metadata: ServiceMetaDataDTO
    tools: list[ToolDTO]


def _tool_to_dto(tool: Tool) -> ToolDTO:
    return ToolDTO(
        id=tool.id,
        creation_utc=tool.creation_utc,
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        required=tool.required,
        consequential=tool.consequential,
    )


def create_router(service_registry: ServiceRegistry) -> APIRouter:
    router = APIRouter()

    def get_service_metadata(name: str, service: ToolService) -> ServiceMetaDataDTO:
        return ServiceMetaDataDTO(
            name=name,
            kind="openapi" if isinstance(service, OpenAPIClient) else "sdk",
            url=getattr(service, "server_url")
            if isinstance(service, OpenAPIClient)
            else getattr(service, "url"),
        )

    @router.put("/{name}")
    async def create_service(
        name: str, request: CreateServiceRequest
    ) -> CreateOpenAPIServiceResponse:
        _ = await service_registry.update_tool_service(
            name=name,
            kind=request.kind,
            url=request.url,
            openapi_json=getattr(request, "openapi_json", None),
        )

        return CreateOpenAPIServiceResponse(name=name)

    @router.delete("/{name}")
    async def delete_service(name: str) -> DeleteServiceResponse:
        await service_registry.delete_service(name)

        return DeleteServiceResponse(name=name)

    @router.get("/")
    async def list_services() -> ListServicesResponse:
        return ListServicesResponse(
            services=[
                get_service_metadata(name, service)
                for name, service in await service_registry.list_tool_services()
            ]
        )

    @router.get("/{name}")
    async def read_service(name: str) -> ServiceDTO:
        service = await service_registry.read_tool_service(name)
        return ServiceDTO(
            metadata=get_service_metadata(name, service),
            tools=[_tool_to_dto(t) for t in await service.list_tools()],
        )

    return router
