from abc import ABC, abstractmethod
from datetime import datetime, timezone
import importlib
import inspect
import json
from typing import Mapping, Optional, Sequence
from pydantic import ValidationError

from emcie.common.tools import ToolId, ToolParameter, Tool
from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.common import JSONSerializable, generate_id
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
    ) -> JSONSerializable: ...


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
    ) -> JSONSerializable:
        try:
            tool_doc = await self._collection.find_one({"id": {"$eq": tool_id}})
            module = importlib.import_module(tool_doc["module_path"])
            func = getattr(module, tool_doc["name"])
        except Exception as e:
            raise ToolImportError(tool_id) from e

        try:
            result: JSONSerializable = func(**arguments)

            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            raise ToolExecutionError(tool_id) from e

        try:
            return json.dumps(result)
        except Exception as e:
            raise ToolResultError(tool_id) from e
