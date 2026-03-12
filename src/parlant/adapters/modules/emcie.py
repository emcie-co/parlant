from contextlib import AsyncExitStack
import os
from typing import Mapping
from typing_extensions import override

from httpx import AsyncClient
from lagom import Container
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from parlant.core.tracer import Tracer, CompositeTracer
from parlant.core.loggers import Logger, CompositeLogger
from parlant.core.meter import Meter
from parlant.adapters.tracing.emcie import EmcieTracer
from parlant.adapters.loggers.emcie import EmcieLogger
from parlant.adapters.meters.emcie import EmcieMeter
from parlant.api.authorization import AuthorizationPolicy, Operation, ProductionAuthorizationPolicy


EXIT_STACK = AsyncExitStack()


class EmcieAuthorizationPolicy(AuthorizationPolicy):
    def __init__(self, trusted_origins: list[str]) -> None:
        self._trusted_origins = [origin.lower().rstrip("/") for origin in trusted_origins]
        self._production_policy = ProductionAuthorizationPolicy()

    async def configure_app(self, app: FastAPI) -> FastAPI:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        return app

    def _is_trusted(self, headers: Mapping[str, str]) -> bool:
        origin = headers.get("origin", "").lower().rstrip("/")
        return origin in self._trusted_origins

    @property
    @override
    def name(self) -> str:
        return "emcie"

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._production_policy.check_permission(request, operation)

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._production_policy.check_rate_limit(request, operation)

    @override
    async def check_websocket_permission(self, websocket: WebSocket, operation: Operation) -> bool:
        if self._is_trusted(websocket.headers):
            return True
        return await self._production_policy.check_websocket_permission(websocket, operation)


async def configure_module(container: Container) -> Container:
    api_key = os.environ.get("EMCIE_API_KEY")
    api_url = os.environ.get("EMCIE_BASE_URL", "https://api.emcie.co")
    tracer = container[Tracer]

    if not api_key:
        return container  # No API key, skip

    auth_url = f"{api_url}/v1/auth/api-key"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with AsyncClient() as client:
            resp = await client.post(auth_url, headers=headers)
            resp.raise_for_status()
    except Exception:
        return container

    # Authenticated: add EmcieTracer to CompositeTracer
    emcie_tracer = await EXIT_STACK.enter_async_context(EmcieTracer())
    if isinstance(tracer, CompositeTracer):
        tracer.append(emcie_tracer)
        # No need to reassign - we mutated the object in place
    else:
        # Create a new CompositeTracer and define it in container
        composite_tracer = CompositeTracer([tracer, emcie_tracer])
        container.define(Tracer, composite_tracer)

    # Add EmcieLogger to CompositeLogger if logger exists in container
    try:
        logger = container[Logger]
        emcie_logger = await EXIT_STACK.enter_async_context(EmcieLogger(tracer=tracer))

        if isinstance(logger, CompositeLogger):
            logger.append(emcie_logger)
            # No need to reassign - we mutated the object in place
        else:
            # Create a new CompositeLogger and define it in container
            composite_logger = CompositeLogger([logger, emcie_logger])
            container.define(Logger, composite_logger)
    except Exception:
        # Logger not in container, skip logger configuration
        pass

    # Add EmcieMeter if no meter exists in container
    try:
        # Check if meter exists by trying to get it
        _ = container[Meter]
        # If it exists, skip - we can't replace it due to Lagom constraints
    except Exception:
        # Meter not in container, add EmcieMeter
        emcie_meter = await EXIT_STACK.enter_async_context(EmcieMeter())
        container[Meter] = emcie_meter

    container[AuthorizationPolicy] = EmcieAuthorizationPolicy(trusted_origins=[api_url])

    return container
