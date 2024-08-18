import asyncio
from contextlib import asynccontextmanager
import json
import httpx
from random import random
from typing import Any, AsyncIterator
from fastapi import Body, FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn

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


async def one_required_body_param(
    body_param: str = Body(),
) -> JSONResponse:
    return JSONResponse({"result": body_param})


async def two_required_body_params(
    body_param_1: str = Body(),
    body_param_2: str = Body(),
) -> JSONResponse:
    return JSONResponse({"result": body_param_1 + body_param_2})


async def one_required_query_param_one_required_body_param(
    query_param: int = Query(),
    body_param: str = Body(),
) -> JSONResponse:
    return JSONResponse({"result": f"{body_param}: {query_param}"})


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
    return json.dumps(await get_json(f"{address}/openapi.json"))


TOOLS = (
    one_required_query_param,
    two_required_query_params,
    one_required_body_param,
    two_required_body_params,
    one_required_query_param_one_required_body_param,
    dto_object,
)


def rng_app() -> FastAPI:
    app = FastAPI()

    for tool in TOOLS:
        app.get(f"/{tool.__name__}", operation_id=tool.__name__)(tool)

    return app


async def test_that_a_tool_is_exposed_via_an_openapi_server() -> None:
    async with run_openapi_server(rng_app()):
        openapi_json = await get_openapi_spec(OPENAPI_SERVER_URL)

        async with OpenAPIClient(OPENAPI_SERVER_URL, openapi_json) as client:
            tools = await client.list_tools()

            for tool_id, tool in {t.__name__: t for t in TOOLS}.items():
                listed_tool = next((t for t in tools if t.id == tool_id), None)
                assert listed_tool
