# Copyright 2024 Emcie Co Ltd.
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

"""Application module for the API package."""

from parlant.api.imports import *
from pathlib import Path

class AppWrapper:
    """Wrapper for the FastAPI application that provides access to the container."""

    def __init__(self, app: FastAPI, container: Container) -> None:
        """Initialize the AppWrapper.

        Args:
            app: The FastAPI application.
            container: The dependency injection container.
        """
        self.app = app
        self.container = container

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Call the FastAPI application.

        Args:
            scope: The ASGI scope.
            receive: The ASGI receive function.
            send: The ASGI send function.
        """
        await self.app(scope, receive, send)


def create_api_app(container: Container) -> AppWrapper:
    """Create the FastAPI application.

    Args:
        container: The dependency injection container.

    Returns:
        The FastAPI application wrapped in an AppWrapper.
    """
    api_app = FastAPI()

    # Set up CORS
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for Sandbox UI
    chat_dist_path = Path(__file__).parent / "chat" / "dist"
    if chat_dist_path.exists():
        # Mount the dist folder at /sandbox
        api_app.mount("/sandbox", StaticFiles(directory=str(chat_dist_path), html=True), name="sandbox")
        
        # Also mount the same folder at /chat for asset references
        api_app.mount("/chat", StaticFiles(directory=str(chat_dist_path), html=True), name="chat")
        
        # Add root route for redirection to sandbox
        @api_app.get("/", include_in_schema=False)
        async def root():
            """Redirect to the sandbox UI."""
            return RedirectResponse(url="/sandbox/index.html")
    else:
        logger.warning(f"Sandbox UI files not found at {chat_dist_path}")
        
        # Fallback root route
        @api_app.get("/", include_in_schema=False)
        async def root():
            """Return a message when sandbox UI is not available."""
            return {"message": "Parlant API Server", "docs_url": "/docs"}

    # Set up logging
    logger = container[Logger]
    ws_logger = container[WebSocketLogger]
    correlator = container[ContextualCorrelator]

    # Set up stores
    agent_store = container[AgentStore]
    customer_store = container[CustomerStore]
    evaluation_store = container[EvaluationStore]
    fragment_store = container[FragmentStore]
    glossary_store = container[GlossaryStore]
    guideline_store = container[GuidelineStore]
    guideline_connection_store = container[GuidelineConnectionStore]
    guideline_tool_association_store = container[GuidelineToolAssociationStore]
    session_store = container[SessionStore]
    tag_store = container[TagStore]
    variable_store = container[ContextVariableStore]

    # Set up services
    nlp_service = container[NLPService]
    service_registry = container[ServiceRegistry]
    behavioral_change_evaluator = container[BehavioralChangeEvaluator]

    # Set up routers
    api_app.include_router(
        index.create_router(
            evaluation_service=behavioral_change_evaluator,
            evaluation_store=evaluation_store,
            evaluation_listener=container[EvaluationListener],
            agent_store=agent_store,
        ),
        prefix="/index"
    )
    api_app.include_router(
        services.create_router(service_registry=service_registry),
        prefix="/services"
    )
    api_app.include_router(
        tags.create_router(tag_store=tag_store),
        prefix="/tags"
    )
    api_app.include_router(
        customers.create_router(customer_store=customer_store),
        prefix="/customers"
    )
    api_app.include_router(
        fragments.create_router(fragment_store=fragment_store),
        prefix="/fragments"
    )
    api_app.include_router(
        glossary.create_router(glossary_store=glossary_store),
        prefix="/glossary"
    )
    api_app.include_router(
        guidelines.create_router(
            application=container[Application],
            guideline_store=guideline_store,
            guideline_connection_store=guideline_connection_store,
            service_registry=service_registry,
            guideline_tool_association_store=guideline_tool_association_store,
        ),
        prefix="/guidelines"
    )
    api_app.include_router(
        variables.create_router(
            context_variable_store=variable_store,
            service_registry=service_registry,
        ),
        prefix="/variables"
    )
    api_app.include_router(
        agents.create_router(agent_store=agent_store),
        prefix="/agents"
    )
    api_app.include_router(
        sessions.create_router(
            logger=logger,
            application=container[Application],
            agent_store=agent_store,
            customer_store=customer_store,
            session_store=session_store,
            session_listener=container[SessionListener],
            nlp_service=nlp_service,
        ),
        prefix="/sessions"
    )
    api_app.include_router(
        logs.create_router(ws_logger),
        prefix="/logs"
    )

    # Set up DSPy routes
    setup_dspy_routes(api_app, container)

    return AppWrapper(api_app, container) 