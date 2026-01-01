# Copyright 2026 Emcie Co Ltd.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import socket
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from random import randint
from time import sleep
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    TypedDict,
    cast,
)
from typing_extensions import override

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
import httpx
from lagom import Container
import uvicorn

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.nlp.openai_service import GPT_4o
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.application import Application
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel, JSONSerializable, Version
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.customers import Customer, CustomerId, CustomerStore
from parlant.core.engines.alpha.hooks import EngineHook, EngineHooks
from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.loggers import LogLevel, Logger
from parlant.core.meter import LocalMeter
from parlant.core.nlp.generation import (
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.services.tools.mcp_service import MCPToolServer
from parlant.core.services.tools.plugins import PluginServer, ToolEntry
from parlant.core.sessions import (
    _GenerationInfoDocument,
    _UsageInfoDocument,
    Event,
    MessageEventData,
    Session,
    SessionId,
    SessionStore,
    EventSource,
    EventKind,
)
from parlant.core.tags import Tag, TagId
from parlant.core.tools import LocalToolService, ToolId, ToolResult
from parlant.core.tracer import LocalTracer
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import BaseDocument, DocumentCollection

T = TypeVar("T")
GLOBAL_SCHEMATIC_GENERATION_CACHE_FILE = Path("schematic_generation_test_cache.json")
GLOBAL_EMBEDDER_CACHE_FILE = Path("schematic_generation_test_cache.json")

SERVER_PORT = 8089
PLUGIN_SERVER_PORT = 8091
OPENAPI_SERVER_PORT = 8092

SERVER_BASE_URL = "http://localhost"
SERVER_ADDRESS = f"{SERVER_BASE_URL}:{SERVER_PORT}"


class NLPTestSchema(DefaultBaseModel):
    reasoning: str | None = None
    answer: bool


class SyncAwaiter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def __call__(self, awaitable: Generator[Any, None, T] | Awaitable[T]) -> T:
        return self.event_loop.run_until_complete(awaitable)  # type: ignore


@dataclass(frozen=False)
class JournalingEngineHooks(EngineHooks):
    latest_context_per_trace_id: dict[str, EngineContext] = field(default_factory=dict)

    @override
    async def call_hooks(
        self,
        hooks: Sequence[EngineHook],
        context: EngineContext,
        payload: Any,
        exc: Optional[Exception] = None,
    ) -> bool:
        self.latest_context_per_trace_id[context.tracer.trace_id] = context
        return await super().call_hooks(hooks, context, payload, exc)


class _TestLogger(Logger):
    def __init__(self) -> None:
        self.logger = logging.getLogger("TestLogger")

    def set_level(self, log_level: LogLevel) -> None:
        self.logger.setLevel(
            {
                LogLevel.TRACE: logging.DEBUG,
                LogLevel.DEBUG: logging.DEBUG,
                LogLevel.INFO: logging.INFO,
                LogLevel.WARNING: logging.WARNING,
                LogLevel.ERROR: logging.ERROR,
                LogLevel.CRITICAL: logging.CRITICAL,
            }[log_level]
        )

    def trace(self, message: str) -> None:
        self.logger.debug(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    @contextmanager
    def scope(self, scope_id: str) -> Iterator[None]:
        yield


async def nlp_test(context: str, condition: str) -> bool:
    schematic_generator = GPT_4o[NLPTestSchema](
        logger=_TestLogger(), tracer=LocalTracer(), meter=LocalMeter(_TestLogger())
    )

    inference = await schematic_generator.generate(
        prompt=f"""\
Given a context and a condition, determine whether the
condition applies with respect to the given context.
If the condition applies, the answer is true;
otherwise, the answer is false.

Context: ###
{context}
###

Condition: ###
{condition}
###

Output JSON structure: ###
{{
    "reasoning": <STRING>,
    "answer": <BOOL>
}}
###

Example #1: ###
{{
    "reasoning": "The condition holds because...",
    "answer": true
}}
###

Example #2: ###
{{
    "reasoning": "The condition doesn't hold because...",
    "answer": false
}}
###
""",
        hints={"temperature": 0.0, "strict": True},
    )
    return inference.content.answer


async def create_agent(container: Container, name: str) -> Agent:
    return await container[AgentStore].create_agent(name="test-agent", max_engine_iterations=2)


async def create_customer(container: Container, name: str) -> Customer:
    return await container[CustomerStore].create_customer(
        name=name,
        extra={"email": "test@customer.com"},
    )


async def create_session(
    container: Container,
    agent_id: AgentId,
    customer_id: Optional[CustomerId] = None,
    title: Optional[str] = None,
    metadata: Optional[Mapping[str, JSONSerializable]] = None,
) -> Session:
    return await container[SessionStore].create_session(
        customer_id or (await create_customer(container, "Auto-Created Customer")).id,
        agent_id=agent_id,
        title=title,
        metadata=metadata or {},
    )


async def create_term(
    container: Container,
    agent_id: AgentId,
    name: str,
    description: str,
    synonyms: list[str],
) -> Term:
    term = await container[GlossaryStore].create_term(
        name=name,
        description=description,
        synonyms=synonyms,
    )

    await container[GlossaryStore].upsert_tag(
        term_id=term.id,
        tag_id=Tag.for_agent_id(agent_id),
    )

    return term


async def create_context_variable(
    container: Container,
    name: str,
    tags: list[TagId],
    description: str = "",
) -> ContextVariable:
    return await container[ContextVariableStore].create_variable(
        name=name,
        description=description,
        tool_id=None,
        freshness_rules=None,
    )


async def set_context_variable_value(
    container: Container,
    variable_id: ContextVariableId,
    key: str,
    data: JSONSerializable,
) -> ContextVariableValue:
    return await container[ContextVariableStore].update_value(
        key=key,
        variable_id=variable_id,
        data=data,
    )


async def create_guideline(
    container: Container,
    agent_id: AgentId,
    condition: str,
    action: str,
    tool_function: Optional[Callable[[], ToolResult]] = None,
) -> Guideline:
    guideline = await container[GuidelineStore].create_guideline(
        condition=condition,
        action=action,
    )

    _ = await container[GuidelineStore].upsert_tag(
        guideline.id,
        Tag.for_agent_id(agent_id),
    )

    if tool_function:
        local_tool_service = container[LocalToolService]

        existing_tools = await local_tool_service.list_tools()

        tool = next(
            (
                t
                for t in existing_tools
                if t.name == getattr(tool_function, "__name__", "unnamed_tool")
            ),
            None,
        )

        if not tool:
            tool = await local_tool_service.create_tool(
                name=getattr(tool_function, "__name__", "unnamed_tool"),
                module_path=tool_function.__module__,
                description="",
                parameters={},
                required=[],
            )

        await container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId("local", getattr(tool_function, "__name__", "unnamed_tool")),
        )

    return guideline


async def read_reply(
    container: Container,
    session_id: SessionId,
    customer_event_offset: int,
) -> Event:
    return next(
        iter(
            await container[SessionStore].list_events(
                session_id=session_id,
                source=EventSource.AI_AGENT,
                min_offset=customer_event_offset,
                kinds=[EventKind.MESSAGE],
            )
        )
    )


async def post_message(
    container: Container,
    session_id: SessionId,
    message: str,
    response_timeout: Timeout = Timeout.none(),
    metadata: Mapping[str, JSONSerializable] | None = None,
) -> Event:
    customer_id = (await container[SessionStore].read_session(session_id)).customer_id
    customer = await container[CustomerStore].read_customer(customer_id)

    data: MessageEventData = {
        "message": message,
        "participant": {
            "id": customer_id,
            "display_name": customer.name,
        },
    }

    event = await container[Application].sessions.create_event(
        session_id=session_id,
        kind=EventKind.MESSAGE,
        data=data,
        metadata=metadata,
    )

    if response_timeout:
        await container[Application].sessions.wait_for_update(
            session_id=session_id,
            min_offset=event.offset + 1,
            kinds=[EventKind.MESSAGE],
            timeout=response_timeout,
        )

    return event


async def get_when_async_done_or_timeout(
    result_getter: Callable[[], Awaitable[T]],
    done_condition: Callable[[T], bool],
    timeout: int,
) -> T:
    for _ in range(timeout):
        result = await result_getter()
        if done_condition(result):
            return result
        await asyncio.sleep(1)

    raise TimeoutError()


def get_when_done_or_timeout(
    result_getter: Callable[[], T],
    done_condition: Callable[[T], bool],
    timeout: int,
) -> T:
    for _ in range(timeout):
        result = result_getter()
        if done_condition(result):
            return result
        sleep(1)

    raise TimeoutError()


TBaseModel = TypeVar("TBaseModel", bound=DefaultBaseModel)


class SchematicGenerationResultDocument(TypedDict, total=False):
    id: ObjectId
    creation_utc: str
    version: Version.String
    content: JSONSerializable
    info: _GenerationInfoDocument


class CachedSchematicGenerator(SchematicGenerator[TBaseModel]):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        base_generator: SchematicGenerator[TBaseModel],
        collection: DocumentCollection[SchematicGenerationResultDocument],
        use_cache: bool,
    ):
        self._base_generator = base_generator
        self._collection = collection
        self.use_cache = use_cache

        self._ensure_cache_file_exists()

    def _ensure_cache_file_exists(self) -> None:
        if not GLOBAL_SCHEMATIC_GENERATION_CACHE_FILE.exists():
            GLOBAL_SCHEMATIC_GENERATION_CACHE_FILE.write_text("{}")

    def _generate_id(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any],
    ) -> str:
        sorted_hints = json.dumps(dict(sorted(hints.items())), sort_keys=True)
        key_content = f"{self.id}:{prompt}:{sorted_hints}"
        return hashlib.sha256(key_content.encode()).hexdigest()

    def _serialize_result(
        self,
        id: str,
        result: SchematicGenerationResult[TBaseModel],
    ) -> SchematicGenerationResultDocument:
        def serialize_generation_info(generation: GenerationInfo) -> _GenerationInfoDocument:
            return _GenerationInfoDocument(
                schema_name=generation.schema_name,
                model=generation.model,
                duration=generation.duration,
                usage=_UsageInfoDocument(
                    input_tokens=generation.usage.input_tokens,
                    output_tokens=generation.usage.output_tokens,
                    extra=generation.usage.extra,
                ),
            )

        return SchematicGenerationResultDocument(
            id=ObjectId(id),
            creation_utc=datetime.now(tz=timezone.utc).isoformat(),
            version=self.VERSION.to_string(),
            content=result.content.model_dump(mode="json"),
            info=serialize_generation_info(result.info),
        )

    def _deserialize_result(
        self,
        doc: SchematicGenerationResultDocument,
        schema_type: type[TBaseModel],
    ) -> SchematicGenerationResult[TBaseModel]:
        def deserialize_generation_info(
            generation_document: _GenerationInfoDocument,
        ) -> GenerationInfo:
            return GenerationInfo(
                schema_name=generation_document["schema_name"],
                model=generation_document["model"],
                duration=generation_document["duration"],
                usage=UsageInfo(
                    input_tokens=generation_document["usage"]["input_tokens"],
                    output_tokens=generation_document["usage"]["output_tokens"],
                    extra=generation_document["usage"]["extra"],
                ),
            )

        content = schema_type.model_validate(doc["content"])
        info = deserialize_generation_info(doc["info"])

        return SchematicGenerationResult[TBaseModel](
            content=content,
            info=info,
        )

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[TBaseModel]:
        if isinstance(prompt, PromptBuilder):
            prompt_text = prompt.build()

        if self.use_cache is False:
            return await self._base_generator.generate(prompt_text, hints)

        id = self._generate_id(prompt_text, hints)

        result_document = await self._collection.find_one(filters={"id": {"$eq": id}})
        if result_document:
            schema_type = (
                self._base_generator.schema
                if type(self._base_generator) is not FallbackSchematicGenerator
                else cast(FallbackSchematicGenerator[TBaseModel], self._base_generator)
                ._generators[0]
                .schema
            )

            return self._deserialize_result(doc=result_document, schema_type=schema_type)

        result = await self._base_generator.generate(prompt, hints)
        await self._collection.insert_one(document=self._serialize_result(id=id, result=result))

        return result

    @property
    def id(self) -> str:
        return self._base_generator.id

    @property
    def max_tokens(self) -> int:
        return self._base_generator.max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._base_generator.tokenizer


@asynccontextmanager
async def create_schematic_generation_result_collection(
    logger: Logger,
) -> AsyncIterator[DocumentCollection[SchematicGenerationResultDocument]]:
    async def _document_loader(doc: BaseDocument) -> Optional[SchematicGenerationResultDocument]:
        if doc["version"] == "0.1.0":
            return cast(SchematicGenerationResultDocument, doc)
        return None

    async with JSONFileDocumentDatabase(logger, GLOBAL_SCHEMATIC_GENERATION_CACHE_FILE) as db:
        yield await db.get_or_create_collection(
            name="schematic_generation_result_cache",
            schema=SchematicGenerationResultDocument,
            document_loader=_document_loader,
        )


@asynccontextmanager
async def run_service_server(
    tools: list[ToolEntry],
    plugin_data: Mapping[str, Any] = {},
) -> AsyncIterator[PluginServer]:
    port = get_random_port(50001, 65535)

    async with PluginServer(
        tools=tools,
        port=port,
        host="127.0.0.1",
        plugin_data=plugin_data,
    ) as server:
        try:
            yield server
        finally:
            await server.shutdown()


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


def rng_app(port: int = OPENAPI_SERVER_PORT) -> FastAPI:
    app = FastAPI(servers=[{"url": f"{SERVER_BASE_URL}:{port}"}])

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


def is_port_available(port: int, host: str = "localhost") -> bool:
    available = True
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)  # Short timeout for faster testing
        sock.bind((host, port))
    except (socket.error, OSError):
        available = False
    finally:
        sock.close()

    return available


def get_random_port(
    min_port: int = 1024,
    max_port: int = 65535,
    max_iterations: int = sys.maxsize,
) -> int:
    iter = 0
    while not is_port_available(port := randint(min_port, max_port)) and iter < max_iterations:
        iter += 1
        pass
    return port


class DummyDTO(DefaultBaseModel):
    number: int
    text: str


async def dto_object(dto: DummyDTO) -> JSONResponse:
    return JSONResponse({})


@dataclass
class ServerInfo:
    port: int
    url: str


@asynccontextmanager
async def run_openapi_server(
    app: Optional[FastAPI] = None,
) -> AsyncIterator[ServerInfo]:
    port = get_random_port(10001, 65535)

    if app is None:
        app = rng_app(port=port)

    config = uvicorn.Config(app=app, port=port)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    try:
        while not server.started:
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.05)

        server_info = ServerInfo(
            port=port,
            url=SERVER_BASE_URL,
        )

        yield server_info
    finally:
        server.should_exit = True
        await asyncio.sleep(0.1)

        # If it's still running close it more aggressively
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@asynccontextmanager
async def run_mcp_server(tools: Sequence[Callable[..., Any]] = []) -> AsyncIterator[ServerInfo]:
    port = get_random_port(10001, 65535)

    server = MCPToolServer(
        port=port,
        host=SERVER_BASE_URL,
        tools=tools,
    )

    try:
        await server.__aenter__()

        # Wait for server to start with timeout
        start_timeout = 8
        sample_frequency = 0.1
        for _ in range(int(start_timeout / sample_frequency)):
            if server.started():
                break
            await asyncio.sleep(sample_frequency)
        else:
            raise TimeoutError("MCP server failed to start within timeout period")

        # Additional wait to ensure server is fully initialized
        await asyncio.sleep(0.5)

        server_info = ServerInfo(
            port=port,
            url=SERVER_BASE_URL,
        )

        yield server_info
    finally:
        try:
            await server.__aexit__(None, None, None)
        except Exception:
            pass


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
