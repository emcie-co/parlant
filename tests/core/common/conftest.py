import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
import uvicorn
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from lagom import Container
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.customers import CustomerId, CustomerStore
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EventEmitter, EventEmitterFactory
from parlant.core.guidelines import GuidelineStore
from parlant.core.services.tools.plugins import PluginClient, PluginServer
from parlant.core.sessions import Session, SessionId, SessionStore
from parlant.core.tools import ToolContext
from pytest import fixture


OPENAPI_SERVER_PORT = 8089
OPENAPI_SERVER_URL = f"http://localhost:{OPENAPI_SERVER_PORT}"

AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING = 0.3
REASONABLE_AMOUNT_OF_TIME = 10


@dataclass
class ApplicationContextOfTest:
    container: Container
    app: Application
    customer_id: CustomerId


@fixture
async def application_context(
    container: Container,
    customer_id: CustomerId,
) -> ApplicationContextOfTest:
    return ApplicationContextOfTest(
        container=container,
        app=container[Application],
        customer_id=customer_id,
    )


@fixture
async def agent_id(container: Container) -> AgentId:
    store = container[AgentStore]
    agent = await store.create_agent(name="test-agent")
    return agent.id


@fixture
async def proactive_agent_id(
    container: Container,
    agent_id: AgentId,
) -> AgentId:
    await container[GuidelineStore].create_guideline(
        guideline_set=agent_id,
        condition="The customer hasn't engaged yet",
        action="Greet the customer",
    )

    return agent_id


@fixture
async def session(
    container: Container,
    customer_id: CustomerId,
    agent_id: AgentId,
) -> Session:
    store = container[SessionStore]
    session = await store.create_session(
        customer_id=customer_id,
        agent_id=agent_id,
    )
    return session


@fixture
async def customer_id(container: Container) -> CustomerId:
    store = container[CustomerStore]
    customer = await store.create_customer("Larry David", extra={"email": "larry@seinfeld.com"})
    return customer.id


async def one_required_query_param(
    query_param: int = Query(),
) -> JSONResponse:
    return JSONResponse({"result": query_param})


async def two_required_query_params(
    query_param_1: int = Query(),
    query_param_2: int = Query(),
) -> JSONResponse:
    return JSONResponse({"result": query_param_1 + query_param_2})


class OneBodyParam(DefaultBaseModel):
    body_param: str


async def one_required_body_param(
    body: OneBodyParam,
) -> JSONResponse:
    return JSONResponse({"result": body.body_param})


class TwoBodyParams(DefaultBaseModel):
    body_param_1: str
    body_param_2: str


async def two_required_body_params(
    body: TwoBodyParams,
) -> JSONResponse:
    return JSONResponse({"result": body.body_param_1 + body.body_param_2})


async def one_required_query_param_one_required_body_param(
    body: OneBodyParam,
    query_param: int = Query(),
) -> JSONResponse:
    return JSONResponse({"result": f"{body.body_param}: {query_param}"})


class DummyDTO(DefaultBaseModel):
    number: int
    text: str


async def dto_object(dto: DummyDTO) -> JSONResponse:
    return JSONResponse({})


@asynccontextmanager
async def run_openapi_server(app: FastAPI) -> AsyncIterator[None]:
    config = uvicorn.Config(app=app, port=OPENAPI_SERVER_PORT)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    yield
    server.should_exit = True
    await task


async def get_json(address: str, params: dict[str, str] = {}) -> Any:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(address, params=params)
        response.raise_for_status()
        return response.json()


async def get_openapi_spec(address: str) -> str:
    return json.dumps(await get_json(f"{address}/openapi.json"), indent=2)


TOOLS = (
    one_required_query_param,
    two_required_query_params,
    one_required_body_param,
    two_required_body_params,
    one_required_query_param_one_required_body_param,
    dto_object,
)


def rng_app() -> FastAPI:
    app = FastAPI(servers=[{"url": OPENAPI_SERVER_URL}])

    @app.middleware("http")
    async def debug_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        return response

    for tool in TOOLS:
        registration_func = app.post if "body" in tool.__name__ else app.get
        registration_func(f"/{tool.__name__}", operation_id=tool.__name__)(tool)

    return app


class SessionBuffers(EventEmitterFactory):
    def __init__(self, agent_store: AgentStore) -> None:
        self.agent_store = agent_store
        self.for_session: dict[SessionId, EventBuffer] = {}

    async def create_event_emitter(
        self,
        emitting_agent_id: AgentId,
        session_id: SessionId,
    ) -> EventEmitter:
        agent = await self.agent_store.read_agent(emitting_agent_id)
        buffer = EventBuffer(emitting_agent=agent)
        self.for_session[session_id] = buffer
        return buffer


@fixture
async def agent(container: Container) -> Agent:
    return await container[AgentStore].create_agent(name="Test Agent")


@fixture
async def customer_context(agent: Agent) -> ToolContext:
    return ToolContext(
        agent_id=agent.id,
        session_id="test_session",
        customer_id="test_customer",
    )


def create_client(
    server: PluginServer,
    event_emitter_factory: EventEmitterFactory,
) -> PluginClient:
    return PluginClient(
        url=server.url,
        event_emitter_factory=event_emitter_factory,
        correlator=ContextualCorrelator(),
    )
