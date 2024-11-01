from typing import Awaitable, Callable
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from lagom import Container

from emcie.server.api import agents, index
from emcie.server.api import sessions
from emcie.server.api import glossary
from emcie.server.api import guidelines
from emcie.server.api import context_variables as variables
from emcie.server.api import services
from emcie.server.core.context_variables import ContextVariableStore
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.agents import AgentStore
from emcie.server.core.common import ItemNotFoundError, generate_id
from emcie.server.core.evaluations import EvaluationStore
from emcie.server.core.guideline_connections import GuidelineConnectionStore
from emcie.server.core.guidelines import GuidelineStore
from emcie.server.core.guideline_tool_associations import GuidelineToolAssociationStore
from emcie.server.core.services.tools.service_registry import ServiceRegistry
from emcie.server.core.sessions import SessionListener, SessionStore
from emcie.server.core.glossary import GlossaryStore
from emcie.server.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
)
from emcie.server.core.logging import Logger
from emcie.server.core.mc import MC


async def create_app(container: Container) -> FastAPI:
    logger = container[Logger]
    correlator = container[ContextualCorrelator]
    agent_store = container[AgentStore]
    session_store = container[SessionStore]
    session_listener = container[SessionListener]
    evaluation_store = container[EvaluationStore]
    evaluation_service = container[BehavioralChangeEvaluator]
    glossary_store = container[GlossaryStore]
    guideline_store = container[GuidelineStore]
    guideline_connection_store = container[GuidelineConnectionStore]
    guideline_tool_association_store = container[GuidelineToolAssociationStore]
    context_variable_store = container[ContextVariableStore]
    service_registry = container[ServiceRegistry]
    mc = container[MC]

    app = FastAPI()

    app.add_middleware(CORSMiddleware, allow_origins=["*"])

    @app.middleware("http")
    async def add_correlation_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        with correlator.correlation_scope(f"request({generate_id()})"):
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

    agent_router = APIRouter()

    agent_router.include_router(
        agents.create_router(
            agent_store=agent_store,
        )
    )

    agent_router.include_router(
        guidelines.create_router(
            mc=mc,
            guideline_store=guideline_store,
            guideline_connection_store=guideline_connection_store,
            service_registry=service_registry,
            guideline_tool_association_store=guideline_tool_association_store,
        )
    )
    agent_router.include_router(
        index.create_router(
            evaluation_service=evaluation_service,
            evaluation_store=evaluation_store,
            agent_store=agent_store,
        )
    )
    agent_router.include_router(
        glossary.create_router(
            glossary_store=glossary_store,
        )
    )
    agent_router.include_router(
        variables.create_router(
            context_variable_store=context_variable_store,
            service_registry=service_registry,
        )
    )

    app.include_router(
        prefix="/agents",
        router=agent_router,
    )

    app.include_router(
        prefix="/sessions",
        router=sessions.create_router(
            mc=mc,
            agent_store=agent_store,
            session_store=session_store,
            session_listener=session_listener,
            service_registry=service_registry,
        ),
    )

    app.include_router(
        prefix="/services",
        router=services.create_router(
            service_registry=service_registry,
        ),
    )

    return app
