from typing import Awaitable, Callable
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from lagom import Container

from emcie.server.api import agents
from emcie.server.api import sessions
from emcie.server.contextual_correlator import ContextualCorrelator
from emcie.server.core.agents import AgentStore
from emcie.server.core.common import ItemNotFoundError, generate_id
from emcie.server.core.sessions import SessionListener, SessionStore
from emcie.server.logger import Logger
from emcie.server.mc import MC


async def create_app(container: Container) -> FastAPI:
    logger = container[Logger]
    correlator = container[ContextualCorrelator]
    agent_store = container[AgentStore]
    session_store = container[SessionStore]
    session_listener = container[SessionListener]
    mc = container[MC]

    app = FastAPI()

    app.add_middleware(CORSMiddleware, allow_origins=["*"])

    @app.middleware("http")
    async def add_correlation_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        with correlator.correlation_scope(generate_id()):
            logger.info(f"{request.method} {request.url.path}")
            return await call_next(request)

    @app.exception_handler(ItemNotFoundError)
    async def item_not_found_error_handler(
        request: Request, exc: ItemNotFoundError
    ) -> HTTPException:
        logger.info(str(exc))

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    app.include_router(
        prefix="/agents",
        router=agents.create_router(
            agent_store=agent_store,
        ),
    )

    app.include_router(
        prefix="/sessions",
        router=sessions.create_router(
            mc=mc,
            session_store=session_store,
            session_listener=session_listener,
        ),
    )

    return app
