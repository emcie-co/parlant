import asyncio
from contextlib import asynccontextmanager
import json
import httpx
from typing import Any, AsyncIterator, Awaitable, Callable
from fastapi import Body, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from pytest import mark
import uvicorn

from emcie.common.tools import ToolId
from emcie.server.core.services.openapi import OpenAPIClient
from emcie.common.base_models import DefaultBaseModel

OPENAPI_SERVER_PORT = 8089
OPENAPI_SERVER_URL = f"http://localhost:{OPENAPI_SERVER_PORT}"


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


async def test_that_tools_are_exposed_via_an_openapi_server() -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        async with OpenAPIClient(OPENAPI_SERVER_URL, openapi_json) as client:
            tools = await client.list_tools()

            for tool_id, tool in {t.__name__: t for t in TOOLS}.items():
                listed_tool = next((t for t in tools if t.id == tool_id), None)
                assert listed_tool


async def test_that_tools_can_be_read_via_an_openapi_server() -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        async with OpenAPIClient(OPENAPI_SERVER_URL, openapi_json) as client:
            tools = await client.list_tools()

            for t in tools:
                assert (await client.read_tool(t.id)) == t


@mark.parametrize(
    ["tool_id", "tool_args", "expected_result"],
    [
        (
            one_required_query_param.__name__,
            {"query_param": 123},
            {"result": 123},
        ),
        (
            two_required_query_params.__name__,
            {"query_param_1": 123, "query_param_2": 321},
            {"result": 123 + 321},
        ),
        (
            one_required_body_param.__name__,
            {"body_param": "hello"},
            {"result": "hello"},
        ),
        (
            two_required_body_params.__name__,
            {"body_param_1": "hello ", "body_param_2": "world"},
            {"result": "hello world"},
        ),
        (
            one_required_query_param_one_required_body_param.__name__,
            {"body_param": "banana", "query_param": 123},
            {"result": "banana: 123"},
        ),
    ],
)
async def test_that_a_tool_can_be_called_via_an_openapi_server(
    tool_id: ToolId,
    tool_args: dict[str, Any],
    expected_result: Any,
) -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        async with OpenAPIClient(OPENAPI_SERVER_URL, openapi_json) as client:
            result = await client.call_tool(tool_id, tool_args)
            assert result.data == expected_result
