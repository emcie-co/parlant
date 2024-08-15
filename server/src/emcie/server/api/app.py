from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lagom import Container

from emcie.server.api import agents
from emcie.server.api import sessions
from emcie.server.core.agents import AgentStore
from emcie.server.core.sessions import SessionListener, SessionStore
from emcie.server.mc import MC


async def create_app(container: Container) -> FastAPI:
    agent_store = container[AgentStore]
    session_store = container[SessionStore]
    session_listener = container[SessionListener]
    mc = container[MC]

    app = FastAPI()

    app.add_middleware(CORSMiddleware, allow_origins=["*"])

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
