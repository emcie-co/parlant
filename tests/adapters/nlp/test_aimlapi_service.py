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

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionChunk
from parlant.adapters.nlp.aimlapi_service import (
    AIMLAPIReactGenerator,
    AIMLAPISchematicGenerator,
    AIMLAPIEmbedder,
    build_attribution_headers,
    register_aimlapi_models,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.container import Container
from parlant.core.loggers import StdoutLogger, Logger
from parlant.core.meter import LocalMeter, Meter
from parlant.core.tracer import LocalTracer, Tracer
from parlant.core.usage_reporter import UsageReporter
from parlant.core.nlp.model_routing import ModelImplementationRegistry, NullSecretLoader
from parlant.core.nlp.react import (
    Message,
    Role,
    TextPart,
    ReasoningConfig,
    TurnBuilder,
    FinishReason,
)


class Answer(DefaultBaseModel):
    answer: str


@pytest.fixture
def services() -> dict[str, Any]:
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    return dict(logger=logger, tracer=tracer, meter=LocalMeter(logger))


def test_that_headers_are_scoped_fresh_and_identify_parlant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PARLANT_CLIENT_IDENTITY", raising=False)
    headers = build_attribution_headers("https://api.aimlapi.com/v1")
    assert headers["X-AIMLAPI-Source"] == "agent/parlant"
    assert headers["X-AIMLAPI-Partner-ID"] == "part_UOT3mCwOdpOUQKX2gIvYrCmv"
    headers.clear()
    assert build_attribution_headers("https://api.aimlapi.com/v1")
    assert build_attribution_headers("https://proxy.example/v1") == {}
    assert build_attribution_headers("https://api.aimlapi.com.evil.example/v1") == {}


def test_that_registration_exposes_aimlapi(services: dict[str, Any]) -> None:
    container = Container()
    container[Logger] = services["logger"]
    container[Tracer] = services["tracer"]
    container[Meter] = services["meter"]
    container[UsageReporter] = UsageReporter(services["tracer"])
    registry = ModelImplementationRegistry(NullSecretLoader())
    register_aimlapi_models(registry, container)
    assert "aimlapi" in registry.providers


def test_that_react_omits_unset_optional_parameters(services: dict[str, Any]) -> None:
    generator = AIMLAPIReactGenerator(
        model="openai/gpt-4.1", logger=services["logger"], api_key="offline"
    )
    request = generator._encode(
        [Message(role=Role.USER, parts=[TextPart(text="Hi")])],
        [],
        "auto",
        reasoning=ReasoningConfig(),
    )
    assert request == {"model": "openai/gpt-4.1", "messages": [{"role": "user", "content": "Hi"}]}


def test_that_stream_reassembles_tool_arguments_and_usage(services: dict[str, Any]) -> None:
    generator = AIMLAPIReactGenerator(
        model="openai/gpt-4.1", logger=services["logger"], api_key="offline"
    )
    builder = TurnBuilder()
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":'},
                }
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]},
    ]
    for delta in deltas:
        generator._decode(
            ChatCompletionChunk(
                id="c",
                created=0,
                model=generator.model,
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": delta, "finish_reason": None}],
            ),
            builder,
        )
    generator._decode(
        ChatCompletionChunk(
            id="c",
            created=0,
            model=generator.model,
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        ),
        builder,
    )
    assert builder.finish_reason == FinishReason.TOOL_CALLS
    assert builder.usage.input_tokens == 12
    assert builder.usage.output_tokens == 3
    assert builder.finish().tool_calls[0].args == {"q": "hello"}


@pytest.mark.asyncio
async def test_that_schematic_validates_json_and_omits_none(services: dict[str, Any]) -> None:
    from openai.types.chat import ChatCompletion

    generator = AIMLAPISchematicGenerator[Answer](
        model_name="openai/gpt-4.1", api_key="offline", **services
    )
    create = AsyncMock(
        return_value=ChatCompletion(
            id="c",
            created=0,
            model="openai/gpt-4.1",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"answer":"yes"}'},
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
    )
    cast(Any, generator._client.chat.completions).create = create
    result = await generator.do_generate("Answer in JSON", {"temperature": None})
    assert result.content.answer == "yes"
    assert result.info.usage.input_tokens == 10
    assert "temperature" not in create.call_args.kwargs
    assert create.call_args.kwargs["response_format"]["type"] == "json_object"
    await generator._client.close()


@pytest.mark.asyncio
async def test_that_embedding_has_correct_dimensions_before_first_request(
    services: dict[str, Any],
) -> None:
    from openai.types import CreateEmbeddingResponse

    embedder = AIMLAPIEmbedder(
        model_name="openai/text-embedding-3-large", api_key="offline", **services
    )
    assert embedder.dimensions == 3072
    create = AsyncMock(
        return_value=CreateEmbeddingResponse(
            model=embedder.model_name,
            object="list",
            data=[{"index": 0, "object": "embedding", "embedding": [0.0] * 3072}],
            usage={"prompt_tokens": 2, "total_tokens": 2},
        )
    )
    cast(Any, embedder._client.embeddings).create = create
    result = await embedder.do_embed(["hello"])
    assert len(result.vectors[0]) == 3072
    assert "dimensions" not in create.call_args.kwargs
    assert result.usage.input_tokens == 2
    await embedder._client.close()


def test_that_missing_key_has_actionable_error(
    services: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from parlant.core.nlp.model_routing import MissingProviderCredentialsError

    monkeypatch.delenv("AIMLAPI_API_KEY", raising=False)
    with pytest.raises(MissingProviderCredentialsError) as error:
        AIMLAPIReactGenerator(model="openai/gpt-4.1", logger=services["logger"])
    assert error.value.env_var == "AIMLAPI_API_KEY"


def test_that_identity_opt_out_removes_partner_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARLANT_CLIENT_IDENTITY", "false")
    assert build_attribution_headers() == {}


@pytest.mark.asyncio
async def test_that_unknown_embedding_dimensions_are_rejected(services: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        AIMLAPIEmbedder(model_name="custom/embedding", api_key="offline", **services)


@pytest.mark.asyncio
async def test_that_invalid_json_schema_is_rejected(services: dict[str, Any]) -> None:
    from openai.types.chat import ChatCompletion
    from pydantic import ValidationError

    generator = AIMLAPISchematicGenerator[Answer](
        model_name="openai/gpt-4.1", api_key="offline", **services
    )
    cast(Any, generator._client.chat.completions).create = AsyncMock(
        return_value=ChatCompletion(
            id="c",
            created=0,
            model="openai/gpt-4.1",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"wrong":true}'},
                }
            ],
        )
    )
    with pytest.raises(ValidationError):
        await generator.do_generate("JSON please")
    await generator._client.close()


@pytest.mark.asyncio
async def test_that_http_request_uses_aimlapi_endpoint_and_headers(
    services: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx2
    import json

    monkeypatch.delenv("PARLANT_CLIENT_IDENTITY", raising=False)
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "c",
                "created": 0,
                "model": "openai/gpt-4.1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"answer":"yes"}'},
                    }
                ],
            },
        )

    generator = AIMLAPISchematicGenerator[Answer](
        model_name="openai/gpt-4.1", api_key="offline", **services
    )
    await generator._client._client.aclose()
    generator._client._client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    await generator.do_generate("Return JSON", {"temperature": None})
    assert str(requests[0].url) == "https://api.aimlapi.com/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer offline"
    assert requests[0].headers["X-AIMLAPI-Source"] == "agent/parlant"
    assert "temperature" not in json.loads(requests[0].content)
    await generator._client.close()


@pytest.mark.asyncio
async def test_that_streaming_http_response_completes_a_react_step(
    services: dict[str, Any],
) -> None:
    import httpx2
    import json

    requests: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        chunks: list[dict[str, Any]] = [
            {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 1,
                    "total_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
        ]
        body = (
            "".join(
                "data: "
                + json.dumps(
                    {
                        "id": "c",
                        "created": 0,
                        "model": "openai/gpt-4.1",
                        "object": "chat.completion.chunk",
                        **chunk,
                    }
                )
                + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    generator = AIMLAPIReactGenerator(
        model="openai/gpt-4.1", logger=services["logger"], api_key="offline"
    )
    await generator._client._client.aclose()
    generator._client._client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    result = await generator.step([Message(role=Role.USER, parts=[TextPart(text="Hi")])], [])
    assert result.message.text == "Hello"
    assert result.usage.input_tokens == 6
    assert result.usage.cached_input_tokens == 2
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert "tools" not in requests[0]
    await generator._client.close()


@pytest.mark.asyncio
async def test_that_http_authentication_errors_propagate(services: dict[str, Any]) -> None:
    import httpx2
    from openai import AuthenticationError

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401, json={"error": {"message": "Invalid key", "type": "authentication_error"}}
        )

    generator = AIMLAPISchematicGenerator[Answer](
        model_name="openai/gpt-4.1", api_key="offline", **services
    )
    await generator._client._client.aclose()
    generator._client._client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    with pytest.raises(AuthenticationError):
        await generator.do_generate("Return JSON")
    await generator._client.close()


@pytest.mark.asyncio
async def test_that_server_loader_constructs_default_capabilities(
    services: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from parlant.bin.server import register_model_implementation_loaders
    from parlant.core.nlp.model_routing import ModelCapability

    monkeypatch.setenv("AIMLAPI_API_KEY", "offline")
    container = Container()
    container[Logger] = services["logger"]
    container[Tracer] = services["tracer"]
    container[Meter] = services["meter"]
    container[UsageReporter] = UsageReporter(services["tracer"])
    registry = ModelImplementationRegistry(NullSecretLoader())
    register_model_implementation_loaders(registry, container)
    for capability, route in [
        (ModelCapability.SCHEMATIC, "matching.test"),
        (ModelCapability.REACT, "react"),
        (ModelCapability.EMBEDDING, "embedding.rules"),
        (ModelCapability.TOKENIZER, "tokenizer"),
    ]:
        config = registry.default_config("aimlapi", capability, route, {})
        model = await registry.construct(
            config=config,
            capability=capability,
            route=route,
            scope="test",
            schema=Answer if capability == ModelCapability.SCHEMATIC else None,
        )
        assert model is not None
    assert not registry.has_default("aimlapi", ModelCapability.MODERATION)
    assert not registry.has_default("aimlapi", ModelCapability.DECISION)


def test_that_invalid_json_is_not_replaced_with_an_empty_object(services: dict[str, Any]) -> None:
    from parlant.adapters.nlp.aimlapi_service import _parse_json_content

    with pytest.raises(ValueError):
        _parse_json_content("This is not JSON", services["logger"])


@pytest.mark.asyncio
async def test_that_embedding_dimension_mismatch_is_rejected(services: dict[str, Any]) -> None:
    from openai.types import CreateEmbeddingResponse

    embedder = AIMLAPIEmbedder(
        model_name="openai/text-embedding-3-large", api_key="offline", **services
    )
    cast(Any, embedder._client.embeddings).create = AsyncMock(
        return_value=CreateEmbeddingResponse(
            model=embedder.model_name,
            object="list",
            data=[{"index": 0, "object": "embedding", "embedding": [0.0]}],
            usage={"prompt_tokens": 2, "total_tokens": 2},
        )
    )
    with pytest.raises(ValueError, match="dimension"):
        await embedder.do_embed(["hello"])
    await embedder._client.close()
