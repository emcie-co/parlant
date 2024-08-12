from abc import ABC, abstractmethod
from datetime import datetime, timezone
import importlib
import inspect
from typing import Mapping, Optional, Sequence
from pydantic import ValidationError

from emcie.common.tools import ToolId, ToolParameter, ToolResult, Tool
from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.common import generate_id
from emcie.server.core.persistence.document_database import DocumentDatabase


class ToolError(Exception):
    def __init__(
        self,
        tool_id: ToolId,
        message: Optional[str] = None,
    ) -> None:
        if message:
            super().__init__(f"Tool error (id='{tool_id}'): {message}")
        else:
            super().__init__(f"Tool error (id='{tool_id}')")

        self.tool_id = tool_id


class ToolImportError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class ToolResultError(ToolError):
    pass


class ToolService(ABC):
    @abstractmethod
    async def list_tools(
        self,
    ) -> Sequence[Tool]: ...

    @abstractmethod
    async def read_tool(
        self,
        tool_id: ToolId,
    ) -> Tool: ...

    @abstractmethod
    async def call_tool(
        self,
        tool_id: ToolId,
        arguments: dict[str, object],
    ) -> ToolResult: ...


class MultiplexedToolService(ToolService):
    def __init__(self, services: Mapping[str, ToolService] = {}) -> None:
        self.services = services

    async def list_tools(self) -> Sequence[Tool]:
        tools = [
            self._multiplex_tool(service_name, t)
            for service_name, service in self.services.items()
            for t in await service.list_tools()
        ]
        return tools

    async def read_tool(self, tool_id: ToolId, service_name: Optional[str] = None) -> Tool:
        if service_name:
            actual_tool_id = str(tool_id)
        else:
            service_name, actual_tool_id = self._demultiplex_tool_str(tool_id)

        service = self.services[service_name]
        tool = await service.read_tool(ToolId(actual_tool_id))
        return self._multiplex_tool(service_name, tool)

    async def call_tool(
        self,
        tool_id: ToolId,
        arguments: dict[str, object],
    ) -> ToolResult:
        service_name, actual_tool_id = self._demultiplex_tool_str(tool_id)
        service = self.services[service_name]
        return await service.call_tool(ToolId(actual_tool_id), arguments)

    def _multiplex_tool(self, service_name: str, tool: Tool) -> Tool:
        return Tool(
            id=ToolId(f"{service_name}__{tool.id}"),
            name=f"{service_name}__{tool.name}",
            creation_utc=tool.creation_utc,
            description=tool.description,
            parameters=tool.parameters,
            required=tool.required,
            consequential=tool.consequential,
        )

    def _demultiplex_tool_str(self, tool_str: str) -> tuple[str, str]:
        service_name = tool_str[: tool_str.find("__")]
        tool_str = tool_str[len(service_name) + 2 :]
        return service_name, tool_str

    def _demultiplex_tool(self, tool: Tool) -> tuple[str, Tool]:
        service_name, tool_id = self._demultiplex_tool_str(tool.id)
        _, tool_name = self._demultiplex_tool_str(tool.name)

        return (
            service_name,
            Tool(
                id=ToolId(tool_id),
                name=tool_name,
                creation_utc=tool.creation_utc,
                description=tool.description,
                parameters=tool.parameters,
                required=tool.required,
                consequential=tool.consequential,
            ),
        )


class LocalToolService(ToolService):
    class ToolDocument(DefaultBaseModel):
        id: ToolId
        creation_utc: datetime
        name: str
        module_path: str
        description: str
        parameters: Mapping[str, ToolParameter]
        required: Sequence[str]
        consequential: bool

    def __init__(
        self,
        database: DocumentDatabase,
    ) -> None:
        self._collection = database.get_or_create_collection(
            name="tools",
            schema=self.ToolDocument,
        )

    async def create_tool(
        self,
        name: str,
        module_path: str,
        description: str,
        parameters: Mapping[str, ToolParameter],
        required: Sequence[str],
        creation_utc: Optional[datetime] = None,
        consequential: bool = False,
    ) -> Tool:
        if list(await self._collection.find(filters={"name": {"$eq": name}})):
            raise ValidationError("Tool name must be unique within the tool set")

        creation_utc = creation_utc or datetime.now(timezone.utc)
        tool_id = await self._collection.insert_one(
            document={
                "id": generate_id(),
                "name": name,
                "module_path": module_path,
                "description": description,
                "parameters": parameters,
                "required": required,
                "creation_utc": creation_utc,
                "consequential": consequential,
            },
        )

        return Tool(
            id=ToolId(tool_id),
            name=name,
            description=description,
            parameters=dict(parameters),
            creation_utc=creation_utc,
            required=list(required),
            consequential=consequential,
        )

    async def list_tools(
        self,
    ) -> Sequence[Tool]:
        return [
            Tool(
                id=ToolId(d["id"]),
                name=d["name"],
                description=d["description"],
                parameters=d["parameters"],
                creation_utc=d["creation_utc"],
                required=d["required"],
                consequential=d["consequential"],
            )
            for d in await self._collection.find(filters={})
        ]

    async def read_tool(
        self,
        tool_id: ToolId,
    ) -> Tool:
        tool_document = await self._collection.find_one(filters={"id": {"$eq": tool_id}})

        return Tool(
            id=ToolId(tool_document["id"]),
            name=tool_document["name"],
            description=tool_document["description"],
            parameters=tool_document["parameters"],
            creation_utc=tool_document["creation_utc"],
            required=tool_document["required"],
            consequential=tool_document["consequential"],
        )

    async def call_tool(
        self,
        tool_id: ToolId,
        arguments: dict[str, object],
    ) -> ToolResult:
        try:
            tool_doc = await self._collection.find_one({"id": {"$eq": tool_id}})
            module = importlib.import_module(tool_doc["module_path"])
            func = getattr(module, tool_doc["name"])
        except Exception as e:
            raise ToolImportError(tool_id) from e

        try:
            result: ToolResult = func(**arguments)

            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            raise ToolExecutionError(tool_id) from e

        if not isinstance(result, ToolResult):
            raise ToolResultError(tool_id, "Tool result is not an instance of ToolResult")

        return result
