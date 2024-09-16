from typing import Awaitable, Callable
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from lagom import Container

from emcie.server.api import agents, index
from emcie.server.api import sessions
from emcie.server.api import terminology
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.agents import AgentStore
from emcie.server.core.common import ItemNotFoundError, generate_id
from emcie.server.core.evaluations import EvaluationStore
from emcie.server.core.sessions import SessionListener, SessionStore
from emcie.server.core.terminology import TerminologyStore
from emcie.server.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
)
from emcie.server.core.logger import Logger
from emcie.server.core.mc import MC


async def create_app(container: Container) -> FastAPI:
    logger = container[Logger]
    correlator = container[ContextualCorrelator]
    agent_store = container[AgentStore]
    session_store = container[SessionStore]
    session_listener = container[SessionListener]
    evaluation_store = container[EvaluationStore]
    evaluation_service = container[BehavioralChangeEvaluator]
    terminology_store = container[TerminologyStore]
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

    app.include_router(
        prefix="/index",
        router=index.create_router(
            evaluation_service=evaluation_service,
            evaluation_store=evaluation_store,
        ),
    )

    app.include_router(
        prefix="/terminology",
        router=terminology.create_router(
            terminology_store=terminology_store,
        ),
    )

    return app
