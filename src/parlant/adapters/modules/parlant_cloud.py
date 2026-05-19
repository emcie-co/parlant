import hmac
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import override

from parlant.api.authorization import AuthorizationPolicy, Operation, ProductionAuthorizationPolicy

PLATFORM_SECRET_HEADER = "X-Parlant-Cloud-Platform-Secret"


class ParlantCloudAuthorizationPolicy(AuthorizationPolicy):
    def __init__(self, platform_secret: str) -> None:
        self._platform_secret = platform_secret
        self._production_policy = ProductionAuthorizationPolicy()

    @property
    @override
    def name(self) -> str:
        return "parlant-cloud"

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
        secret = headers.get(PLATFORM_SECRET_HEADER, "")
        return bool(secret) and hmac.compare_digest(secret, self._platform_secret)

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
