from datetime import datetime
from typing import Optional, Union, cast
from fastapi import APIRouter
from typing_extensions import Literal

from emcie.server.core.common import DefaultBaseModel
from emcie.server.core.tools import Tool, ToolParameter, ToolParameterType
from emcie.server.core.services.tools.openapi import OpenAPIClient
from emcie.server.core.services.tools.plugins import PluginClient
from emcie.server.core.services.tools.service_registry import ServiceRegistry
from emcie.server.core.tools import ToolService


class CreateSDKServiceRequest(DefaultBaseModel):
    kind: Literal["sdk"]
    url: str


class CreateOpenAPIServiceRequest(DefaultBaseModel):
    kind: Literal["openapi"]
    url: str
    source: str


CreateServiceRequest = Union[CreateSDKServiceRequest, CreateOpenAPIServiceRequest]

ToolServiceKind = Literal["openapi", "sdk"]


class CreateServiceResponse(DefaultBaseModel):
    name: str
    kind: ToolServiceKind
    url: str


class DeleteServiceResponse(DefaultBaseModel):
    name: str


class ToolParameterDTO(DefaultBaseModel):
    type: ToolParameterType
    description: Optional[str]
    enum: Optional[list[Union[str, int, float, bool]]]


class ToolDTO(DefaultBaseModel):
    creation_utc: datetime
    name: str
    description: str
    parameters: dict[str, ToolParameterDTO]
    required: list[str]


class ServiceDTO(DefaultBaseModel):
    name: str
    kind: ToolServiceKind
    url: str
    tools: Optional[list[ToolDTO]] = None


class ListServicesResponse(DefaultBaseModel):
    services: list[ServiceDTO]


def _tool_parameters_to_dto(parameters: ToolParameter) -> ToolParameterDTO:
    return ToolParameterDTO(
        type=parameters["type"],
        description=parameters["description"] if "description" in parameters else None,
        enum=parameters["enum"] if "enum" in parameters else None,
    )


def _tool_to_dto(tool: Tool) -> ToolDTO:
    return ToolDTO(
        creation_utc=tool.creation_utc,
        name=tool.name,
        description=tool.description,
        parameters={k: _tool_parameters_to_dto(p) for k, p in tool.parameters.items()},
        required=tool.required,
    )


def _get_service_kind(service: ToolService) -> ToolServiceKind:
    return "openapi" if isinstance(service, OpenAPIClient) else "sdk"


def _get_service_url(service: ToolService) -> str:
    return (
        service.server_url
        if isinstance(service, OpenAPIClient)
        else cast(PluginClient, service).url
    )


def create_router(service_registry: ServiceRegistry) -> APIRouter:
    router = APIRouter()

    @router.put("/{name}")
    async def create_service(name: str, request: CreateServiceRequest) -> CreateServiceResponse:
        service = await service_registry.update_tool_service(
            name=name,
            kind=request.kind,
            url=request.url,
            source=getattr(request, "source", None),
        )

        return CreateServiceResponse(
            name=name,
            kind=_get_service_kind(service),
            url=_get_service_url(service),
        )

    @router.delete("/{name}")
    async def delete_service(name: str) -> DeleteServiceResponse:
        await service_registry.delete_service(name)

        return DeleteServiceResponse(name=name)

    @router.get("/")
    async def list_services() -> ListServicesResponse:
        return ListServicesResponse(
            services=[
                ServiceDTO(
                    name=name,
                    kind=_get_service_kind(service),
                    url=_get_service_url(service),
                )
                for name, service in await service_registry.list_tool_services()
                if type(service) in [OpenAPIClient, PluginClient]
            ]
        )

    @router.get("/{name}")
    async def read_service(name: str) -> ServiceDTO:
        service = await service_registry.read_tool_service(name)

        return ServiceDTO(
            name=name,
            kind=_get_service_kind(service),
            url=_get_service_url(service),
            tools=[_tool_to_dto(t) for t in await service.list_tools()],
        )

    return router
