from __future__ import annotations
import asyncio
from types import TracebackType
from typing import Optional, Type
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from emcie.common.tools import Tool


class ListToolsResponse(BaseModel):
    tools: list[Tool]


class PluginServer:
    def __init__(
        self,
        tools: list[Tool],
        port: int = 8089,
        host: str = "0.0.0.0",
    ) -> None:
        self.tools = tools
        self.host = host
        self.port = port

        self._server: Optional[uvicorn.Server] = None

    async def __aenter__(self) -> PluginServer:
        self._task = asyncio.create_task(self.serve())

        start_timeout = 5
        sample_frequency = 0.1

        for _ in range(int(start_timeout / sample_frequency)):
            await asyncio.sleep(sample_frequency)

            if self.started():
                return self

        raise TimeoutError()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        try:
            await self._task
        except asyncio.CancelledError:
            pass

        return False

    async def serve(self) -> None:
        app = self._create_app()

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

        self._server = uvicorn.Server(config)

        await self._server.serve()

    async def shutdown(self) -> None:
        self._task.cancel()

    def started(self) -> bool:
        if self._server:
            return self._server.started
        return False

    def _create_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/tools")
        async def list_tools() -> ListToolsResponse:
            return ListToolsResponse(tools=self.tools)

        return app
