from typing import Sequence

from parlant.core.loggers import Logger
from parlant.core.services.tools.service_registry import ServiceRegistry, ToolServiceKind
from parlant.core.tools import ToolService


class ServiceModule:
    def __init__(
        self,
        logger: Logger,
        service_registry: ServiceRegistry,
    ):
        self._logger = logger
        self._service_registry = service_registry

    async def read(self, name: str) -> ToolService:
        service = await self._service_registry.read_tool_service(name)
        return service

    async def update(
        self,
        name: str,
        kind: ToolServiceKind,
        url: str,
        source: str | None,
    ) -> ToolService:
        service = await self._service_registry.update_tool_service(
            name=name,
            kind=kind,
            url=url,
            source=source,
        )

        return service

    async def delete(self, name: str) -> None:
        await self._service_registry.read_tool_service(name)
        await self._service_registry.delete_service(name)

    async def find(self) -> Sequence[tuple[str, ToolService]]:
        return await self._service_registry.list_tool_services()
