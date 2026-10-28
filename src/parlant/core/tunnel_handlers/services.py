# Copyright 2026 Emcie Co Ltd.
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

from typing import Any

from parlant.core.app_modules.services import ServiceModule
from parlant.core.common import ItemNotFoundError
from parlant.core.services.tools.mcp_service import MCPToolClient
from parlant.core.services.tools.openapi import OpenAPIClient
from parlant.core.services.tools.plugins import PluginClient
from parlant.core.services.tools.service_registry import ToolServiceKind
from parlant.core.tools import Tool, ToolParameterDescriptor, ToolService
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


class ServiceTunnelHandlers:
    def __init__(self, service_module: ServiceModule) -> None:
        self._service_module = service_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "services.update": self.update,
                "services.list": self.list,
                "services.retrieve": self.retrieve,
                "services.delete": self.delete,
            },
            streams={},
        )

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        name = _require_non_empty_string(params.get("name"), field="name")
        kind = _parse_service_kind(params.get("kind"))
        configuration = _require_object(params.get(kind), field=kind)
        url = _parse_service_url(configuration.get("url"))
        source = (
            _require_string(configuration.get("source"), field="source")
            if kind == "openapi"
            else None
        )

        service = await self._service_module.update(
            name=name,
            kind=kind,
            url=url,
            source=source,
        )
        return {"service": await serialize_service(name, service)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        services = await self._service_module.find()
        return {
            "services": [
                await serialize_service(name, service)
                for name, service in services
                if isinstance(service, (OpenAPIClient, PluginClient, MCPToolClient))
            ]
        }

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        name = _require_non_empty_string(params.get("name"), field="name")
        try:
            service = await self._service_module.read(name=name)
        except ItemNotFoundError:
            return {"service": None}
        return {"service": await serialize_service(name, service)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        name = _require_non_empty_string(params.get("name"), field="name")
        await self._service_module.delete(name=name)
        return {}


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _require_non_empty_string(value: object, *, field: str) -> str:
    parsed = _require_string(value, field=field)
    if not parsed:
        raise ValueError(f"{field} must be a non-empty string")
    return parsed


def _require_object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _parse_service_kind(value: object) -> ToolServiceKind:
    kind = _require_non_empty_string(value, field="kind")
    public_kinds: dict[str, ToolServiceKind] = {
        "sdk": "sdk",
        "openapi": "openapi",
        "mcp": "mcp",
    }
    if kind not in public_kinds:
        raise ValueError(f"Unknown service kind: {kind}")
    return public_kinds[kind]


def _parse_service_url(value: object) -> str:
    url = _require_non_empty_string(value, field="url")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("url must start with http:// or https://")
    return url


def _service_kind_and_url(service: ToolService) -> tuple[ToolServiceKind, str]:
    if isinstance(service, OpenAPIClient):
        return "openapi", service.server_url
    if isinstance(service, PluginClient):
        return "sdk", service.url
    if isinstance(service, MCPToolClient):
        return "mcp", service.endpoint_url
    raise ValueError(f"Unknown service kind: {type(service)}")


async def serialize_service(name: str, service: ToolService) -> dict[str, Any]:
    kind, url = _service_kind_and_url(service)
    tools = await service.list_tools()
    return {
        "name": name,
        "kind": kind,
        "url": url,
        "tools": [serialize_tool(tool) for tool in tools],
    }


def serialize_tool(tool: Tool) -> dict[str, Any]:
    return {
        "creation_utc": tool.creation_utc.isoformat(),
        "name": tool.name,
        "narration": tool.narration
        if tool.narration is None or isinstance(tool.narration, str)
        else list(tool.narration),
        "description": tool.description,
        "parameters": {
            name: serialize_tool_parameter(descriptor)
            for name, (descriptor, _) in tool.parameters.items()
        },
        "required": list(tool.required),
    }


def serialize_tool_parameter(descriptor: ToolParameterDescriptor) -> dict[str, Any]:
    return {
        "type": serialize_tool_parameter_type(descriptor["type"]),
        "description": descriptor.get("description"),
        "enum": list(descriptor["enum"]) if "enum" in descriptor else None,
    }


def serialize_tool_parameter_type(value: object) -> str:
    public_types = {
        "string": "string",
        "number": "number",
        "integer": "integer",
        "boolean": "boolean",
        "array": "array",
    }
    if not isinstance(value, str) or value not in public_types:
        raise ValueError(f"Unknown tool parameter type: {value}")
    return public_types[value]
