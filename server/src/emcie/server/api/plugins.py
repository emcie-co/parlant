from typing import Union
from fastapi import APIRouter
from typing_extensions import Literal

from emcie.server.core.common import DefaultBaseModel
from emcie.server.core.services.tools.service_registry import ServiceRegistry


class CreateSDKPluginRequest(DefaultBaseModel):
    kind: Literal["sdk"]
    url: str


class CreateOpenAPIPluginRequest(DefaultBaseModel):
    kind: Literal["openapi"]
    url: str
    openapi_json: str


CreatePluginRequest = Union[CreateSDKPluginRequest, CreateOpenAPIPluginRequest]


class CreateSDKPluginResponse(DefaultBaseModel):
    name: str
    kind: Literal["sdk"] = "sdk"
    url: str


class CreateOpenAPIPluginResponse(DefaultBaseModel):
    name: str


class DeletePluginResponse(DefaultBaseModel):
    name: str


def create_router(service_registry: ServiceRegistry) -> APIRouter:
    router = APIRouter()

    @router.put("/{name}")
    async def create_plugin(name: str, request: CreatePluginRequest) -> CreateOpenAPIPluginResponse:
        _ = await service_registry.update_tool_service(
            name=name,
            kind=request.kind,
            url=request.url,
            openapi_json=getattr(request, "openapi_json", None),
        )

        return CreateOpenAPIPluginResponse(name=name)

    @router.delete("/{name}", response_model=DeletePluginResponse)
    async def delete_plugin(name: str) -> DeletePluginResponse:
        await service_registry.delete_service(name)

        return DeletePluginResponse(name=name)

    return router
