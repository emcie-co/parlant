# Copyright 2025 Emcie Co Ltd.
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

from enum import Enum
from typing import Annotated, Optional, Sequence, TypeAlias, cast
from fastapi import APIRouter, HTTPException, Path, Request, status
from pydantic import Field

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    ToolDTO,
    apigen_config,
    ExampleJson,
    ServiceNameField,
    tool_to_dto,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.services.tools.mcp_service import MCPToolClient
from parlant.core.services.tools.plugins import PluginClient
from parlant.core.services.tools.openapi import OpenAPIClient
from parlant.core.services.tools.service_registry import ServiceRegistry, ToolServiceKind
from parlant.core.tools import ToolService

API_GROUP = "services"


class ToolServiceKindDTO(Enum):
    """
    The type of service integration available in the system.

    Attributes:
        "sdk": Native integration using the Parlant SDK protocol. Enables advanced features
            like bidirectional communication and streaming results.
        "openapi": Integration via OpenAPI specification. Simpler to set up but limited
            to basic request/response patterns.
        "mcp": Integration with tool servers using the popular MCP (Model Context Protocol)
            implemented by wide variety of 3rd parties.
    """

    SDK = "sdk"
    OPENAPI = "openapi"
    MCP = "mcp"


ServiceParamsURLField: TypeAlias = Annotated[
    str,
    Field(
        description="Base URL for the service. Must include http:// or https:// scheme.",
        examples=["https://example.com/api/v1"],
    ),
]


sdk_service_params_example: ExampleJson = {"url": "https://email-service.example.com/api/v1"}


class SDKServiceParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": sdk_service_params_example},
):
    """
    Configuration parameters for SDK-based service integration.

    SDK services must implement the Parlant SDK protocol for advanced features
    like streaming and bidirectional communication.
    """

    url: ServiceParamsURLField


ServiceOpenAPIParamsSourceField: TypeAlias = Annotated[
    str,
    Field(
        description="""URL or filesystem path to the OpenAPI specification.
        For URLs, must be publicly accessible.
        For filesystem paths, the server must have read permissions.""",
        examples=["https://api.example.com/openapi.json", "/etc/parlant/specs/example-api.yaml"],
    ),
]


openapi_service_params_example: ExampleJson = {
    "url": "https://email-service.example.com/api/v1",
    "source": "https://email-service.example.com/api/openapi.json",
}


class OpenAPIServiceParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": openapi_service_params_example},
):
    """
    Configuration parameters for OpenAPI-based service integration.

    OpenAPI services are integrated using their OpenAPI/Swagger specification,
    enabling automatic generation of client code and documentation.
    """

    url: ServiceParamsURLField
    source: ServiceOpenAPIParamsSourceField


class MCPServiceParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": sdk_service_params_example},
):
    """
    Configuration parameters for MCP-based service integration.

    MCP services use the MCP protocol, which enables advanced features
    and supports a wide variety of variable types. It is widely adopted by third parties worldwide.
    """

    url: ServiceParamsURLField


ServiceUpdateSDKServiceParamsField: TypeAlias = Annotated[
    SDKServiceParamsDTO,
    Field(
        description="SDK service configuration parameters. Required when kind is 'sdk'.",
    ),
]

ServiceUpdateOpenAPIServiceParamsField: TypeAlias = Annotated[
    OpenAPIServiceParamsDTO,
    Field(
        description="OpenAPI service configuration parameters. Required when kind is 'openapi'.",
    ),
]

ServiceUpdateMCPServiceParamsField: TypeAlias = Annotated[
    MCPServiceParamsDTO,
    Field(
        description="MCP service configuration parameters. Required when kind is 'mcp'.",
    ),
]


service_update_params_example: ExampleJson = {
    "kind": "openapi",
    "openapi": {
        "url": "https://email-service.example.com/api/v1",
        "source": "https://email-service.example.com/api/openapi.json",
    },
}


class ServiceUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": service_update_params_example},
):
    """
    Parameters for creating or updating a service integration.

    The appropriate params field (sdk or openapi) must be provided based on the
    service kind. Service tools become temporarily unavailable during updates
    and reconnect automatically.
    """

    kind: ToolServiceKindDTO
    sdk: Optional[ServiceUpdateSDKServiceParamsField] = None
    openapi: Optional[ServiceUpdateOpenAPIServiceParamsField] = None
    mcp: Optional[ServiceUpdateMCPServiceParamsField] = None


ServiceURLField: TypeAlias = Annotated[
    str,
    Field(
        description="Base URL where the service is hosted",
        examples=["https://api.example.com/v1", "https://email-service.internal:8080"],
    ),
]

ServiceToolsField: TypeAlias = Annotated[
    Sequence[ToolDTO],
    Field(
        default=None,
        description="List of tools provided by this service. Only included when retrieving a specific service.",
    ),
]


service_example: ExampleJson = {
    "name": "email-service",
    "kind": "openapi",
    "url": "https://email-service.example.com/api/v1",
    "tools": [
        {
            "creation_utc": "2024-03-24T12:00:00Z",
            "name": "send_email",
            "description": "Sends an email to specified recipients with configurable priority",
            "parameters": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"},
                "priority": {
                    "type": "string",
                    "description": "Priority level for the email",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["to", "subject", "body"],
        }
    ],
}


class ServiceDTO(
    DefaultBaseModel,
    json_schema_extra={"example": service_example},
):
    """
    Details about an integrated service and its available tools.

    Services can be either SDK-based for advanced features or OpenAPI-based
    for simpler integrations. The tools list is only included when retrieving
    a specific service, not in list operations.
    """

    name: ServiceNameField
    kind: ToolServiceKindDTO
    url: ServiceURLField
    tools: Optional[ServiceToolsField] = None


def _get_service_kind(service: ToolService) -> ToolServiceKindDTO:
    if isinstance(service, OpenAPIClient):
        return ToolServiceKindDTO.OPENAPI
    if isinstance(service, PluginClient):
        return ToolServiceKindDTO.SDK
    if isinstance(service, MCPToolClient):
        return ToolServiceKindDTO.MCP
    raise ValueError(f"Unknown service kind: {type(service)}")


def _get_service_url(service: ToolService) -> str:
    if isinstance(service, OpenAPIClient):
        return service.server_url
    if isinstance(service, PluginClient):
        return service.url
    if isinstance(service, MCPToolClient):
        return f"{service.url}:{service.port}"
    raise ValueError(f"Unknown service kind: {type(service)}")


def _tool_service_kind_dto_to_tool_service_kind(dto: ToolServiceKindDTO) -> ToolServiceKind:
    return cast(
        ToolServiceKind,
        {
            ToolServiceKindDTO.OPENAPI: "openapi",
            ToolServiceKindDTO.SDK: "sdk",
            ToolServiceKindDTO.MCP: "mcp",
        }[dto],
    )


def _tool_service_kind_to_dto(kind: ToolServiceKind) -> ToolServiceKindDTO:
    return {
        "openapi": ToolServiceKindDTO.OPENAPI,
        "sdk": ToolServiceKindDTO.SDK,
        "mcp": ToolServiceKindDTO.MCP,
    }[kind]


ServiceNamePath: TypeAlias = Annotated[
    str,
    Path(
        description="Unique identifier for the service",
        examples=["email-service", "payment-processor"],
    ),
]


def create_router(
    authorization_policy: AuthorizationPolicy,
    service_registry: ServiceRegistry,
) -> APIRouter:
    """
    Creates a router instance for service-related operations.

    The router provides endpoints for managing service integrations,
    including both SDK and OpenAPI based services. It handles service
    registration, updates, and querying available tools.
    """
    router = APIRouter()

    @router.put(
        "/{name}",
        operation_id="update_service",
        response_model=ServiceDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Service successfully created or updated. The service may take a few seconds to become fully operational as it establishes connections.",
                "content": {"application/json": {"example": service_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "No service found with the given name"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Invalid service configuration parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create_or_update"),
    )
    async def update_service(
        request: Request,
        name: ServiceNamePath,
        params: ServiceUpdateParamsDTO,
    ) -> ServiceDTO:
        """
        Creates a new service or updates an existing one.

        For SDK services:
        - Target server must implement the Parlant SDK protocol
        - Supports bidirectional communication and streaming

        For OpenAPI services:
        - Spec must be accessible and compatible with OpenAPI 3.0
        - Limited to request/response patterns

        Common requirements:
        - Service names must be unique and kebab-case
        - URLs must include http:// or https:// scheme
        - Updates cause brief service interruption while reconnecting
        """
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_SERVICE)

        if params.kind == ToolServiceKindDTO.SDK:
            if not params.sdk:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Missing SDK parameters",
                )

            if not (params.sdk.url.startswith("http://") or params.sdk.url.startswith("https://")):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Service URL is missing schema (http:// or https://)",
                )
        elif params.kind == ToolServiceKindDTO.OPENAPI:
            if not params.openapi:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Missing OpenAPI parameters",
                )
            if not (
                params.openapi.url.startswith("http://")
                or params.openapi.url.startswith("https://")
            ):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Service URL is missing schema (http:// or https://)",
                )
        elif params.kind == ToolServiceKindDTO.MCP:
            if not params.mcp:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Missing MCP parameters",
                )
            if not (params.mcp.url.startswith("http://") or params.mcp.url.startswith("https://")):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Service URL is missing schema (http:// or https://)",
                )
        else:
            raise Exception("Should never logically get here")

        if params.kind == ToolServiceKindDTO.SDK:
            assert params.sdk
            url = params.sdk.url
            source = None
        elif params.kind == ToolServiceKindDTO.OPENAPI:
            assert params.openapi
            url = params.openapi.url
            source = params.openapi.source
        elif params.kind == ToolServiceKindDTO.MCP:
            assert params.mcp
            url = params.mcp.url
            source = None
        else:
            raise Exception("Should never logically get here")

        service = await service_registry.update_tool_service(
            name=name,
            kind=_tool_service_kind_dto_to_tool_service_kind(params.kind),
            url=url,
            source=source,
        )

        return ServiceDTO(
            name=name,
            kind=_get_service_kind(service),
            url=_get_service_url(service),
        )

    @router.delete(
        "/{name}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_service",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Service successfully removed. Any active connections are terminated."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Service not found. May have been deleted by another request."
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_service(
        request: Request,
        name: ServiceNamePath,
    ) -> None:
        """
        Removes a service integration.

        Effects:
        - Active connections are terminated immediately
        - Service tools become unavailable to agents
        - Historical data about tool usage is preserved
        - Running operations may fail
        """
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_SERVICE)

        await service_registry.read_tool_service(name)

        await service_registry.delete_service(name)

    @router.get(
        "",
        operation_id="list_services",
        response_model=Sequence[ServiceDTO],
        responses={
            status.HTTP_200_OK: {
                "description": """List of all registered services. Tool lists are not
                included for performance - use the retrieve endpoint to get tools
                for a specific service.""",
                "content": {"application/json": {"example": [service_example]}},
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_services(request: Request) -> Sequence[ServiceDTO]:
        """
        Returns basic info about all registered services.

        For performance reasons, tool details are omitted from the response.
        Use the retrieve endpoint to get complete information including
        tools for a specific service.
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_SERVICES)

        return [
            ServiceDTO(
                name=name,
                kind=_get_service_kind(service),
                url=_get_service_url(service),
            )
            for name, service in await service_registry.list_tool_services()
            if type(service) in [OpenAPIClient, PluginClient, MCPToolClient]
        ]

    @router.get(
        "/{name}",
        operation_id="read_service",
        response_model=ServiceDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Service details including all available tools",
                "content": {"application/json": {"example": service_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "Service not found"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "Service is registered but currently unavailable"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_service(
        request: Request,
        name: ServiceNamePath,
    ) -> ServiceDTO:
        """
        Get details about a specific service including all its tools.

        The response includes:
        - Basic service information (name, kind, URL)
        - Complete list of available tools
        - Parameter definitions for each tool

        Notes:
        - Tools list may be empty if service is still initializing
        - Parameters marked as required must be provided when using a tool
        - Enum parameters restrict inputs to the listed values
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_SERVICE)

        service = await service_registry.read_tool_service(name)

        return ServiceDTO(
            name=name,
            kind=_get_service_kind(service),
            url=_get_service_url(service),
            tools=[tool_to_dto(t) for t in await service.list_tools()],
        )

    return router
