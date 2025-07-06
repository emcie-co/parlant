# Copyright 2025 Emcie Co Ltd.
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

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import os
from typing import (
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    TypeAlias,
)
from exceptiongroup import ExceptionGroup
from typing_extensions import Self

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send
from lagom import Container

from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.api import agents, capabilities
from parlant.api import evaluations
from parlant.api import index
from parlant.api import journeys
from parlant.api import relationships
from parlant.api import sessions
from parlant.api import glossary
from parlant.api import guidelines
from parlant.api import context_variables as variables
from parlant.api import services
from parlant.api import tags
from parlant.api import customers
from parlant.api import logs
from parlant.api import utterances
from parlant.core.capabilities import CapabilityStore
from parlant.api import guideline_matcher_test_api
from parlant.api import tool_call_inference_test_api
from parlant.core.context_variables import ContextVariableStore
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import AgentStore
from parlant.core.common import ItemNotFoundError, generate_id
from parlant.core.customers import CustomerStore
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import GuidelineMatcher
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolCaller
from parlant.core.evaluations import EvaluationStore, EvaluationListener
from parlant.core.journeys import JourneyStore
from parlant.core.utterances import UtteranceStore
from parlant.core.relationships import RelationshipStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.nlp.service import NLPService
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import SessionListener, SessionStore
from parlant.core.glossary import GlossaryStore
from parlant.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
    LegacyBehavioralChangeEvaluator,
)
from parlant.core.loggers import LogLevel, Logger
from parlant.core.application import Application
from parlant.core.tags import TagStore

ASGIApplication: TypeAlias = Callable[
    [
        Scope,
        Receive,
        Send,
    ],
    Awaitable[None],
]

ASGIApplicationContextManager: TypeAlias = AsyncContextManager[ASGIApplication]

APIConfigurationStep: TypeAlias = Callable[[FastAPI, Container], AsyncContextManager[FastAPI]]

APIConfigurationSteps: TypeAlias = list[APIConfigurationStep]


class AppWrapper:
    def __init__(self, app: FastAPI, container: Container) -> None:
        self.app = app
        self.container = container
        self.stack: Optional[AsyncExitStack] = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """FastAPI's built-in exception handling doesn't catch BaseExceptions
        such as asyncio.CancelledError. This causes the server process to terminate
        with an ugly traceback. This wrapper addresses that by specifically allowing
        asyncio.CancelledError to gracefully exit.
        """

        if self.stack is None:
            raise Exception("attempting to call on app before it was configured.")

        try:
            await self.app(scope, receive, send)

        except asyncio.CancelledError:
            pass

    async def __aenter__(self) -> Self:
        try:
            logger = self.container[Logger]
        except Exception:
            logger = None
        configuration_steps = self.container[APIConfigurationSteps]
        self.stack = AsyncExitStack()

        try:
            for step in configuration_steps:
                configuration_context = step(self.app, self.container)
                await self.stack.enter_async_context(configuration_context)

            return self
        except Exception:
            if logger:
                logger.error("encountered error during configuration step setup")
            await self.__aexit__(None, None, None)
            self.stack = None
            raise

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        try:
            logger = self.container[Logger]
        except Exception:
            logger = None
        if self.stack:
            teardown_exceptions: list[Exception] = []
            exit_callbacks = list(getattr(self.stack, "_exit_callbacks"))
            for cb_index, (_, exit_callback) in enumerate(reversed(exit_callbacks)):
                try:
                    await exit_callback(exc_type, exc_value, traceback)
                except Exception as teardown_exception:
                    teardown_exceptions.append(teardown_exception)
                    if logger:
                        logger.error(f"exception during teardown: {teardown_exception}")

            self.stack.pop_all()

            if exc_type is None and len(teardown_exceptions) > 0:
                raise ExceptionGroup(
                    f"exceptions during teardown [count=({len(teardown_exceptions)}]):",
                    teardown_exceptions,
                )

        return False


@asynccontextmanager
async def configure_middlewares(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    logger = container[Logger]
    correlator = container[ContextualCorrelator]

    @app.middleware("http")
    async def handle_cancellation(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        try:
            return await call_next(request)
        except asyncio.CancelledError:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.middleware("http")
    async def add_correlation_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path.startswith("/chat/"):
            return await call_next(request)

        request_id = generate_id()
        with correlator.correlation_scope(f"RID({request_id})"):
            with logger.operation(
                f"HTTP Request: {request.method} {request.url.path}",
                level=LogLevel.DEBUG,
            ):
                return await call_next(request)

    yield app


@asynccontextmanager
async def configure_cors_middleware(app: FastAPI, _: Container) -> AsyncIterator[FastAPI]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    yield app


@asynccontextmanager
async def configure_exception_handlers(
    app: FastAPI,
    container: Container,
) -> AsyncIterator[FastAPI]:
    logger = container[Logger]

    @app.exception_handler(ItemNotFoundError)
    async def item_not_found_error_handler(
        request: Request, exc: ItemNotFoundError
    ) -> HTTPException:
        logger.info(str(exc))

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    yield app


@asynccontextmanager
async def configure_static_files(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    static_dir = os.path.join(os.path.dirname(__file__), "chat/dist")
    app.mount("/chat", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/", include_in_schema=False)
    async def root() -> Response:
        return RedirectResponse("/chat")

    yield app


@asynccontextmanager
async def configure_legacy_agents(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    agent_router = APIRouter()

    agent_router.include_router(
        guidelines.create_legacy_router(
            application=container[Application],
            guideline_store=container[GuidelineStore],
            relationship_store=container[RelationshipStore],
            service_registry=container[ServiceRegistry],
            tag_store=container[TagStore],
            guideline_tool_association_store=container[GuidelineToolAssociationStore],
        ),
    )

    agent_router.include_router(
        glossary.create_legacy_router(
            glossary_store=container[GlossaryStore],
        )
    )

    agent_router.include_router(
        variables.create_legacy_router(
            context_variable_store=container[ContextVariableStore],
            service_registry=container[ServiceRegistry],
        )
    )

    app.include_router(agent_router, prefix="/agents")
    yield app


@asynccontextmanager
async def configure_agents_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = agents.create_router(
        agent_store=container[AgentStore],
        tag_store=container[TagStore],
    )
    app.include_router(router, prefix="/agents")
    yield app


@asynccontextmanager
async def configure_sessions_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = sessions.create_router(
        logger=container[Logger],
        application=container[Application],
        agent_store=container[AgentStore],
        customer_store=container[CustomerStore],
        session_store=container[SessionStore],
        session_listener=container[SessionListener],
        nlp_service=container[NLPService],
    )
    app.include_router(router, prefix="/sessions")
    yield app


@asynccontextmanager
async def configure_index_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = index.legacy_create_router(
        evaluation_service=container[LegacyBehavioralChangeEvaluator],
        evaluation_store=container[EvaluationStore],
        evaluation_listener=container[EvaluationListener],
        agent_store=container[AgentStore],
    )
    app.include_router(router, prefix="/index")
    yield app


@asynccontextmanager
async def configure_services_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = services.create_router(
        service_registry=container[ServiceRegistry],
    )
    app.include_router(router, prefix="/services")
    yield app


@asynccontextmanager
async def configure_tags_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = tags.create_router(
        tag_store=container[TagStore],
    )
    app.include_router(router, prefix="/tags")
    yield app


@asynccontextmanager
async def configure_terms_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = glossary.create_router(
        glossary_store=container[GlossaryStore],
        agent_store=container[AgentStore],
        tag_store=container[TagStore],
    )
    app.include_router(router, prefix="/terms")
    yield app


@asynccontextmanager
async def configure_customers_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = customers.create_router(
        customer_store=container[CustomerStore],
        tag_store=container[TagStore],
        agent_store=container[AgentStore],
    )
    app.include_router(router, prefix="/customers")
    yield app


@asynccontextmanager
async def configure_utterances_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = utterances.create_router(
        utterance_store=container[UtteranceStore],
        tag_store=container[TagStore],
    )
    app.include_router(router, prefix="/utterances")
    yield app


@asynccontextmanager
async def configure_context_variables_router(
    app: FastAPI, container: Container
) -> AsyncIterator[FastAPI]:
    router = variables.create_router(
        context_variable_store=container[ContextVariableStore],
        service_registry=container[ServiceRegistry],
        agent_store=container[AgentStore],
        tag_store=container[TagStore],
    )
    app.include_router(router, prefix="/context-variables")
    yield app


@asynccontextmanager
async def configure_guidelines_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = guidelines.create_router(
        guideline_store=container[GuidelineStore],
        relationship_store=container[RelationshipStore],
        service_registry=container[ServiceRegistry],
        guideline_tool_association_store=container[GuidelineToolAssociationStore],
        agent_store=container[AgentStore],
        tag_store=container[TagStore],
        journey_store=container[JourneyStore],
    )
    app.include_router(router, prefix="/guidelines")
    yield app


@asynccontextmanager
async def configure_relationships_router(
    app: FastAPI, container: Container
) -> AsyncIterator[FastAPI]:
    router = relationships.create_router(
        guideline_store=container[GuidelineStore],
        tag_store=container[TagStore],
        agent_store=container[AgentStore],
        journey_store=container[JourneyStore],
        relationship_store=container[RelationshipStore],
        service_registry=container[ServiceRegistry],
    )
    app.include_router(router, prefix="/relationships")
    yield app


@asynccontextmanager
async def configure_journeys_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = journeys.create_router(
        journey_store=container[JourneyStore],
        guideline_store=container[GuidelineStore],
    )
    app.include_router(router, prefix="/journeys")
    yield app


@asynccontextmanager
async def configure_evaluations_router(
    app: FastAPI, container: Container
) -> AsyncIterator[FastAPI]:
    router = evaluations.create_router(
        evaluation_service=container[BehavioralChangeEvaluator],
        evaluation_store=container[EvaluationStore],
        evaluation_listener=container[EvaluationListener],
    )
    app.include_router(router, prefix="/evaluations")
    yield app


@asynccontextmanager
async def configure_capabilities_router(
    app: FastAPI, container: Container
) -> AsyncIterator[FastAPI]:
    router = capabilities.create_router(
        capability_store=container[CapabilityStore],
        tag_store=container[TagStore],
        agent_store=container[AgentStore],
        journey_store=container[JourneyStore],
    )
    app.include_router(router, prefix="/capabilities")
    yield app


@asynccontextmanager
async def configure_logs_router(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    router = logs.create_router(
        websocket_logger=container[WebSocketLogger],
    )
    app.include_router(router)
    yield app


@asynccontextmanager
async def configure_test_router(
    app: FastAPI,
    container: Container,
) -> AsyncIterator[FastAPI]:
    test_router_guideline_matching = (
        guideline_matcher_test_api.create_test_guideline_matching_router(
            agent_store=container[AgentStore],
            customer_store=container[CustomerStore],
            context_variable_store=container[ContextVariableStore],
            session_store=container[SessionStore],
            glossary_store=container[GlossaryStore],
            guideline_store=container[GuidelineStore],
            guideline_matcher=container[GuidelineMatcher],
        )
    )
    app.include_router(test_router_guideline_matching, prefix="/test/alpha/guideline-matching")

    test_router_tool_call_inference = (
        tool_call_inference_test_api.create_test_tool_call_inference_router(
            agent_store=container[AgentStore],
            customer_store=container[CustomerStore],
            context_variable_store=container[ContextVariableStore],
            session_store=container[SessionStore],
            glossary_store=container[GlossaryStore],
            guideline_store=container[GuidelineStore],
            service_registry=container[ServiceRegistry],
            journey_store=container[JourneyStore],
            tool_caller=container[ToolCaller],
            logger=container[Logger],
        )
    )
    app.include_router(test_router_tool_call_inference, prefix="/test/alpha/tool-call-inference")

    yield app


default_configuration_steps: APIConfigurationSteps = [
    configure_middlewares,
    configure_cors_middleware,
    configure_exception_handlers,
    configure_static_files,
    configure_legacy_agents,
    configure_agents_router,
    configure_sessions_router,
    configure_index_router,
    configure_services_router,
    configure_tags_router,
    configure_terms_router,
    configure_customers_router,
    configure_utterances_router,
    configure_context_variables_router,
    configure_guidelines_router,
    configure_relationships_router,
    configure_journeys_router,
    configure_evaluations_router,
    configure_logs_router,
    configure_capabilities_router,
]


async def create_api_app(container: Container) -> ASGIApplicationContextManager:
    return AppWrapper(FastAPI(), container)
