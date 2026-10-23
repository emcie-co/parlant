"""Argon behind the existing ReAct contract, without engine-side glue.

Offline protocol cases use a real local HTTP/SSE server and the real Google
client. They simulate provider protocol edge cases, not customer services or
SDK methods. Live cases require GEMINI_API_KEY and use synthetic questions.
Run these protocol cases against google-genai >=2.32.0 during the red stage;
the dependency-floor update belongs to the later implementation stage.
"""

import asyncio
import json
import os
import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import google.genai
import google.genai.types
import httpx
import pytest

from parlant.adapters.nlp.gemini_service import GeminiReactGenerator
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.nlp.react import (
    FinishReason,
    Message,
    ParameterSpec,
    ReactError,
    ReasoningConfig,
    ReasoningDelta,
    Role,
    StepCompleted,
    TextDelta,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    ToolMessageDeserializer,
    ToolMessageSerializer,
    ToolResultPart,
    ToolSpec,
)
from parlant.core.tracer import LocalTracer

MODEL = "gemini-4-argon"
LIVE = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="requires live Gemini")
WEATHER = ToolSpec(
    name="get_weather",
    description="Get weather for a city.",
    parameters=[ParameterSpec(name="city", type="string", description="City name")],
)


@pytest.fixture
def logger() -> Logger:
    return StdoutLogger(LocalTracer())


@dataclass
class Reply:
    events: list[dict[str, Any]] = field(default_factory=list)
    status: int = 200
    hold_open: bool = False


class Provider:
    def __init__(self) -> None:
        self.replies: deque[Reply] = deque()
        self.requests: list[dict[str, Any]] = []
        self.respond: Callable[[dict[str, Any]], Reply] | None = None
        self.release = threading.Event()
        self.responses: list[httpx.Response] = []
        self.lock = threading.Lock()

    async def response_seen(self, response: httpx.Response) -> None:
        self.responses.append(response)

    def client(self, base_url: str) -> google.genai.Client:
        return google.genai.Client(
            api_key="local-protocol-test",
            http_options=google.genai.types.HttpOptions(
                base_url=base_url,
                api_version="v1beta",
                timeout=5000,
                retry_options=google.genai.types.HttpRetryOptions(attempts=1),
                async_client_args={"event_hooks": {"response": [self.response_seen]}},
            ),
        )


@pytest.fixture
def provider() -> Iterator[tuple[Provider, str]]:
    state = Provider()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            with state.lock:
                state.requests.append({"path": self.path, "body": body})
                if not self.path.rstrip("/").endswith("/interactions"):
                    reply = Reply(status=404)
                elif state.respond:
                    reply = state.respond(body)
                else:
                    reply = state.replies.popleft() if state.replies else Reply(status=500)
            self.send_response(reply.status)
            if reply.status != 200:
                data = json.dumps(
                    {"error": {"message": "Synthetic provider failure", "code": reply.status}}
                ).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for event in reply.events:
                    self.wfile.write(
                        f"event: {event['event_type']}\ndata: {json.dumps(event)}\n\n".encode()
                    )
                    self.wfile.flush()
                if reply.hold_open:
                    state.release.wait(timeout=10)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def terminal(status: str = "completed", **usage: int) -> dict[str, Any]:
    return {
        "event_type": "interaction.completed",
        "interaction": {
            "id": "",
            "object": "interaction",
            "model": MODEL,
            "status": status,
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 5,
                "total_thought_tokens": 7,
                "total_cached_tokens": 20,
                "total_tokens": 112,
                **usage,
            },
        },
    }


def text_events(text: str, index: int = 0) -> list[dict[str, Any]]:
    return [
        {"event_type": "step.start", "index": index, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": index, "delta": {"type": "text", "text": text}},
        {"event_type": "step.stop", "index": index},
    ]


def tool_events() -> list[dict[str, Any]]:
    # Shape checked against a real Argon stream captured on 2026-10-21.
    # Signatures here are explicit synthetic protocol fixtures; live tests below
    # verify that actual provider signatures can be replayed successfully.
    events: list[dict[str, Any]] = [
        {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "thought_signature", "signature": "dGhvdWdodA=="},
        },
        {"event_type": "step.stop", "index": 0},
    ]
    for index, city in ((1, "Paris"), (2, "Rome")):
        events.extend(
            [
                {
                    "event_type": "step.start",
                    "index": index,
                    "step": {
                        "type": "function_call",
                        "id": f"call_{index}",
                        "name": "get_weather",
                        "arguments": {},
                        "signature": "c2lnbmF0dXJl",
                    },
                },
                {
                    "event_type": "step.delta",
                    "index": index,
                    "delta": {"type": "arguments_delta", "arguments": '{"city":"'},
                },
                {
                    "event_type": "step.delta",
                    "index": index,
                    "delta": {"type": "arguments_delta", "arguments": city + '"}'},
                },
                {"event_type": "step.stop", "index": index},
            ]
        )
    return [*events, terminal("requires_action")]


def generator_for(logger: Logger, provider: tuple[Provider, str]) -> GeminiReactGenerator:
    state, url = provider
    return GeminiReactGenerator(model=MODEL, logger=logger, client=state.client(url))


def history(question: str = "Hello") -> list[Message]:
    return [Message(role=Role.USER, parts=[TextPart(text=question)])]


@pytest.mark.parametrize("tool_choice", ["auto", "none", "required"])
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [Message(role=Role.SYSTEM, parts=[TextPart(text="Greet the customer.")])],
        history(),
        [Message(role=Role.ASSISTANT, parts=[TextPart(text="Hello!")])],
    ],
    ids=["empty", "system-only", "customer", "assistant-first"],
)
def test_that_argon_uses_stateless_interactions_without_changing_its_model_identity(
    logger: Logger, tool_choice: Any, messages: list[Message]
) -> None:
    generator = GeminiReactGenerator(
        model=MODEL, logger=logger, client=google.genai.Client(api_key="offline")
    )
    original = deepcopy(messages)
    request = generator._encode(messages, [WEATHER], tool_choice, reasoning=ReasoningConfig())
    assert generator.id == "gemini/gemini-4-argon"
    assert generator.provider_name == "gemini"
    assert request["model"] == MODEL
    assert request["store"] is False
    assert request["stream"] is True
    assert not request.get("previous_interaction_id")
    assert not request.get("background")
    assert messages == original
    assert request["generation_config"]["tool_choice"] == {"required": "any"}.get(
        tool_choice, tool_choice
    )


@pytest.mark.parametrize("assistant_tail", [False, True])
def test_that_history_edits_and_mid_turn_instructions_reach_the_provider(
    logger: Logger, assistant_tail: bool
) -> None:
    generator = GeminiReactGenerator(
        model=MODEL, logger=logger, client=google.genai.Client(api_key="offline")
    )
    messages = [
        Message(role=Role.SYSTEM, parts=[TextPart(text="Stable rules")]),
        *history("Edited request"),
        Message(role=Role.SYSTEM, parts=[TextPart(text="Internal review: don't quote a price")]),
    ]
    if assistant_tail:
        messages.append(Message(role=Role.ASSISTANT, parts=[TextPart(text="Let me check.")]))
    original = deepcopy(messages)
    request = generator._encode(messages, [], "none", reasoning=ReasoningConfig())
    encoded = json.dumps(request, default=str)
    assert "Edited request" in encoded
    assert "Stable rules" in encoded
    assert "don't quote a price" in encoded
    assert messages == original
    assert request["input"][-1]["type"] not in {"model_output", "thought", "function_call"}


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
@pytest.mark.parametrize("visibility", ["none", "summary", "full"])
def test_that_argon_thinking_uses_supported_levels_and_a_compatible_output_budget(
    logger: Logger, effort: Any, expected: str, visibility: Any
) -> None:
    generator = GeminiReactGenerator(
        model=MODEL, logger=logger, client=google.genai.Client(api_key="offline")
    )
    request = generator._encode(
        history(), [], "none", reasoning=ReasoningConfig(effort=effort, visibility=visibility)
    )
    config = request["generation_config"]
    assert config["thinking_level"] == expected
    assert config["thinking_summaries"] == ("none" if visibility == "none" else "auto")
    assert config["max_output_tokens"] >= 32768
    assert "thinking_budget" not in config


async def test_that_interaction_text_and_reasoning_deltas_preserve_order_without_duplicates(
    logger: Logger, provider: tuple[Provider, str]
) -> None:
    state, _ = provider
    state.replies.append(
        Reply(
            events=[
                {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
                {
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {
                        "type": "thought_summary",
                        "content": {"type": "text", "text": "Checking"},
                    },
                },
                {"event_type": "step.stop", "index": 0},
                *text_events("Hello", 1),
                {
                    "event_type": "step.start",
                    "index": 2,
                    "step": {"type": "model_output", "content": [{"type": "text", "text": "!"}]},
                },
                {"event_type": "step.stop", "index": 2},
                terminal(),
            ]
        )
    )
    generator = generator_for(logger, provider)
    events = [event async for event in generator.stream_step(history())]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello!"
    assert "".join(e.text for e in events if isinstance(e, ReasoningDelta)) == "Checking"
    completed = [e.result for e in events if isinstance(e, StepCompleted)]
    assert len(completed) == 1
    assert completed[0].message.text == "Hello!"
    assert completed[0].message.reasoning == "Checking"
    assert all(r["path"].endswith("/interactions") for r in state.requests)


async def test_that_split_function_arguments_and_parallel_calls_keep_their_ids(
    logger: Logger, provider: tuple[Provider, str]
) -> None:
    state, _ = provider
    state.replies.append(Reply(events=tool_events()))
    generator = generator_for(logger, provider)
    events = [event async for event in generator.stream_step(history(), [WEATHER])]
    calls = [event for event in events if isinstance(event, ToolCallStarted)]
    assert [(c.id, c.name) for c in calls] == [("call_1", "get_weather"), ("call_2", "get_weather")]
    final = next(e.result for e in events if isinstance(e, StepCompleted))
    assert [dict(call.args) for call in final.tool_calls] == [{"city": "Paris"}, {"city": "Rome"}]
    assert final.finish_reason == FinishReason.TOOL_CALLS


class ToolTurnStore(ToolMessageSerializer, ToolMessageDeserializer):
    def __init__(self) -> None:
        self.calls: list[ToolCallPart] = []
        self.results: list[ToolResultPart] = []
        self.data: dict[str, Any] = {}

    def write_calls(self, calls: Sequence[ToolCallPart]) -> None:
        self.calls = deepcopy(list(calls))

    def write_results(self, results: Sequence[ToolResultPart]) -> None:
        self.results = deepcopy(list(results))

    def write_provider_data(self, data: Mapping[str, Any]) -> None:
        self.data = json.loads(json.dumps(data))

    def read_calls(self) -> Sequence[ToolCallPart]:
        return self.calls

    def read_results(self) -> Sequence[ToolResultPart]:
        return self.results

    def read_provider_data(self) -> Mapping[str, Any]:
        return self.data


@pytest.mark.parametrize("reverse_results", [False, True])
async def test_that_serialized_tool_turns_replay_native_steps_without_reexecuting_tools(
    logger: Logger, provider: tuple[Provider, str], reverse_results: bool
) -> None:
    state, _ = provider
    state.replies.extend(
        [Reply(events=tool_events()), Reply(events=[*text_events("18 degrees"), terminal()])]
    )
    generator = generator_for(logger, provider)
    first = await generator.step(history(), [WEATHER])
    store = ToolTurnStore()
    results = [
        ToolResultPart(
            call_id=c.id, name=c.name, content={"temperature_c": 18, "city": c.args["city"]}
        )
        for c in first.tool_calls
    ]
    generator.serialize_tool_messages(
        [first.message, Message(role=Role.TOOL, parts=results)], store
    )
    replayed = generator.deserialize_tool_messages(store, model=MODEL)
    assert replayed is not None
    assert generator.deserialize_tool_messages(store, model="gemini-3.8-flash") is None
    if reverse_results:
        # Persistence's calls/results port is index-aligned by contract. Reorder
        # only the provider-bound history, where call IDs identify each result.
        replayed = [replayed[0], Message(role=Role.TOOL, parts=list(reversed(replayed[1].parts)))]
    final = await generator.step(
        [*history(), *replayed, *history("Summarize the saved result, don't check again.")],
        [WEATHER],
        tool_choice="none",
    )
    assert not final.tool_calls
    assert "18" in final.message.text
    assert len(state.requests) == 2
    body = state.requests[-1]["body"]
    function_calls = [s for s in body["input"] if s["type"] == "function_call"]
    function_results = [s for s in body["input"] if s["type"] == "function_result"]
    assert {s["id"] for s in function_calls} == {s["call_id"] for s in function_results}
    assert {s["id"]: s["arguments"]["city"] for s in function_calls} == {
        s["call_id"]: json.loads(s["result"][0]["text"])["city"] for s in function_results
    }
    encoded = json.dumps(body)
    assert "dGhvdWdodA==" in encoded
    assert "c2lnbmF0dXJl" in encoded


async def test_that_usage_counts_cached_and_reasoning_tokens_once(
    logger: Logger, provider: tuple[Provider, str]
) -> None:
    state, _ = provider
    done = terminal()
    state.replies.append(Reply(events=[*text_events("Hello"), done, done]))
    result = await generator_for(logger, provider).step(history())
    assert result.usage.input_tokens == 100
    assert result.usage.cached_input_tokens == 20
    assert result.usage.output_tokens == 12
    assert result.usage.reasoning_tokens == 7
    assert result.usage.model_name == MODEL


@pytest.mark.parametrize("ending", ["incomplete", "failed", "truncated", "malformed_arguments"])
async def test_that_incomplete_failed_and_truncated_streams_do_not_look_successful(
    logger: Logger, provider: tuple[Provider, str], ending: str
) -> None:
    state, _ = provider
    if ending == "malformed_arguments":
        events = tool_events()
        for event in events:
            if event.get("delta", {}).get("type") == "arguments_delta":
                event["delta"]["arguments"] = "{not valid JSON"
    else:
        events = [*text_events("Partial"), *([] if ending == "truncated" else [terminal(ending)])]
    state.replies.append(Reply(events=events))
    try:
        result = await generator_for(logger, provider).step(history())
    except ReactError:
        return
    assert result.finish_reason in {FinishReason.MAX_TOKENS, FinishReason.ERROR}


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (401, False), (429, True), (503, True)]
)
async def test_that_interactions_errors_preserve_the_existing_retry_classification(
    logger: Logger, provider: tuple[Provider, str], status: int, retryable: bool
) -> None:
    state, _ = provider
    state.replies.extend(Reply(status=status) for _ in range(5))
    with pytest.raises(ReactError) as error:
        await asyncio.wait_for(generator_for(logger, provider).step(history()), timeout=10)
    assert error.value.retryable is retryable
    assert state.requests[0]["path"].endswith("/interactions")


async def test_that_cancelled_interactions_close_the_stream(
    logger: Logger, provider: tuple[Provider, str]
) -> None:
    state, _ = provider
    state.replies.append(Reply(events=text_events("Hello"), hold_open=True))
    first = asyncio.Event()

    async def consume() -> None:
        async for event in generator_for(logger, provider).stream_step(history()):
            if isinstance(event, TextDelta):
                first.set()

    task = asyncio.create_task(consume())
    started = asyncio.create_task(first.wait())
    try:
        done, _ = await asyncio.wait(
            [task, started], timeout=3, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
        assert first.is_set(), "no content arrived before cancellation"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.responses and state.responses[-1].is_closed
    finally:
        state.release.set()
        task.cancel()
        started.cancel()
        await asyncio.gather(task, started, return_exceptions=True)


async def test_that_concurrent_interactions_keep_their_steps_and_usage_isolated(
    logger: Logger, provider: tuple[Provider, str]
) -> None:
    state, _ = provider
    state.respond = lambda body: Reply(
        events=[*text_events("Alpha" if "Alpha" in json.dumps(body) else "Beta"), terminal()]
    )
    generator = generator_for(logger, provider)
    results = await asyncio.gather(
        generator.step(history("Alpha")), generator.step(history("Beta"))
    )
    assert [result.message.text for result in results] == ["Alpha", "Beta"]
    assert [result.usage.output_tokens for result in results] == [12, 12]
    assert len(state.requests) == 2


def live_generator(logger: Logger) -> tuple[GeminiReactGenerator, list[str], google.genai.Client]:
    paths: list[str] = []

    async def observe(response: httpx.Response) -> None:
        paths.append(response.request.url.path)

    client = google.genai.Client(
        http_options=google.genai.types.HttpOptions(
            async_client_args={"event_hooks": {"response": [observe]}}
        )
    )
    return GeminiReactGenerator(model=MODEL, logger=logger, client=client), paths, client


@LIVE
async def test_that_live_argon_streams_text_and_accounts_for_usage(logger: Logger) -> None:
    generator, paths, client = live_generator(logger)
    try:
        events = [
            event
            async for event in generator.stream_step(
                history("What is six times seven? Answer only the number."),
                tool_choice="none",
                reasoning=ReasoningConfig(effort="low"),
            )
        ]
        assert (
            "".join(e.text for e in events if isinstance(e, TextDelta)).strip().rstrip(".") == "42"
        )
        final = next(e.result for e in events if isinstance(e, StepCompleted))
        assert final.usage.input_tokens > 0
        assert final.usage.output_tokens > 0
        assert final.usage.output_tokens >= final.usage.reasoning_tokens
        assert paths and all(path.endswith("/interactions") for path in paths)
    finally:
        await generator.aclose()
        await client.aio.aclose()


# The engine numbers staged retriever results from "1", so two can share an id.
@LIVE
async def test_that_live_argon_answers_from_tool_results_without_native_calls(
    logger: Logger,
) -> None:
    generator, _, client = live_generator(logger)
    try:
        final = await generator.step(
            [
                *history("What colour are chongas, and what do they taste of?"),
                Message(
                    role=Role.TOOL,
                    parts=[
                        ToolResultPart(
                            call_id="1",
                            name="built-in:rule-retriever-colour",
                            content="Chongas are bright purple",
                        )
                    ],
                ),
                Message(
                    role=Role.TOOL,
                    parts=[
                        ToolResultPart(
                            call_id="1",
                            name="built-in:rule-retriever-taste",
                            content="Chongas taste of mango",
                        )
                    ],
                ),
            ],
            tool_choice="none",
        )
        assert "purple" in final.message.text.lower()
        assert "mango" in final.message.text.lower()
    finally:
        await generator.aclose()
        await client.aio.aclose()


@LIVE
async def test_that_live_argon_runs_parallel_tools_and_continues_after_their_results(
    logger: Logger,
) -> None:
    generator, paths, client = live_generator(logger)
    dispatches: list[str] = []

    async def dispatch(call: ToolCallPart) -> ToolResultPart:
        city = str(call.args["city"])
        dispatches.append(city)
        return ToolResultPart(
            call_id=call.id, name=call.name, content={"city": city, "temperature_c": 18}
        )

    try:
        messages = await asyncio.wait_for(
            generator.run(
                history(
                    "Use get_weather for both Paris and Rome, then tell me both temperatures. Don't check either city twice."
                ),
                [WEATHER],
                dispatch,
                max_steps=4,
            ),
            timeout=120,
        )
        assert len(dispatches) == 2
        assert {city.lower() for city in dispatches} == {"paris", "rome"}
        assert "18" in messages[-1].text
        assert paths and all(path.endswith("/interactions") for path in paths)
    finally:
        await generator.aclose()
        await client.aio.aclose()


@LIVE
async def test_that_live_argon_replays_a_serialized_tool_turn_in_a_later_request(
    logger: Logger,
) -> None:
    generator, paths, client = live_generator(logger)
    try:
        first = await generator.step(
            history("Use get_weather for Paris."),
            [WEATHER],
            tool_choice="required",
            reasoning=ReasoningConfig(effort="low"),
        )
        assert first.tool_calls
        store = ToolTurnStore()
        generator.serialize_tool_messages(
            [
                first.message,
                Message(
                    role=Role.TOOL,
                    parts=[
                        ToolResultPart(
                            call_id=c.id,
                            name=c.name,
                            content={"city": "Paris", "temperature_c": 18},
                        )
                        for c in first.tool_calls
                    ],
                ),
            ],
            store,
        )
        replayed = generator.deserialize_tool_messages(store, model=MODEL)
        assert replayed is not None
        final = await generator.step(
            [
                *history("Use get_weather for Paris."),
                *replayed,
                *history(
                    "What temperature did we find? Use the saved result without checking again."
                ),
            ],
            [WEATHER],
            tool_choice="none",
            reasoning=ReasoningConfig(effort="low"),
        )
        assert not final.tool_calls
        assert "18" in final.message.text
        assert paths and all(path.endswith("/interactions") for path in paths)
    finally:
        await generator.aclose()
        await client.aio.aclose()


async def test_that_retriever_results_without_native_calls_are_sent_as_function_results(
    logger: Logger, provider: tuple[Provider, str]
) -> None:
    # Interactions rejects a synthetic function_call but accepts a
    # function_result that answers no call, so neither a call nor a user-text
    # copy of the result is sent.
    state, _ = provider
    state.replies.append(
        Reply(events=[*text_events("Saved knowledge says 18 degrees"), terminal()])
    )
    messages = [
        *history("What did our knowledge lookup find?"),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(call_id="1", name="knowledge_lookup", content={"temperature_c": 18})
            ],
        ),
    ]
    original = deepcopy(messages)
    final = await generator_for(logger, provider).step(messages, tool_choice="none")
    assert "18" in final.message.text
    steps = state.requests[0]["body"]["input"]
    assert not any(s["type"] == "function_call" for s in steps)
    results = [s for s in steps if s["type"] == "function_result"]
    assert len(results) == 1
    assert results[0]["name"] == "knowledge_lookup"
    assert "18" in json.dumps(results[0]["result"])
    assert not any(s["type"] == "user_input" and "18" in json.dumps(s) for s in steps)
    assert messages == original


@pytest.mark.parametrize("with_system_update", [False, True])
async def test_that_split_assistant_turns_start_with_their_native_thoughts(
    logger: Logger, provider: tuple[Provider, str], with_system_update: bool
) -> None:
    state, _ = provider
    state.replies.extend(
        [Reply(events=tool_events()), Reply(events=[*text_events("18"), terminal()])]
    )
    generator = generator_for(logger, provider)
    first = await generator.step(history(), [WEATHER])
    # Persisted engine history can have the visible preamble as its own event;
    # native call artifacts belong to a separate tool event's assistant message.
    messages = [
        *history(),
        Message(role=Role.ASSISTANT, parts=[TextPart(text="Let me look into that for a minute.")]),
        *(
            [Message(role=Role.SYSTEM, parts=[TextPart(text="Continue following the rules.")])]
            if with_system_update
            else []
        ),
        Message(role=Role.ASSISTANT, parts=list(first.tool_calls)),
        Message(
            role=Role.TOOL,
            parts=[
                ToolResultPart(call_id=call.id, name=call.name, content={"temperature_c": 18})
                for call in first.tool_calls
            ],
        ),
    ]
    original = deepcopy(messages)
    await generator.step(messages, [WEATHER], tool_choice="none")
    encoded = state.requests[-1]["body"]
    assistant_steps = [
        s for s in encoded["input"] if s["type"] in {"model_output", "thought", "function_call"}
    ]
    assert assistant_steps[0]["type"] == "thought"
    assert sum(s["type"] == "thought" for s in assistant_steps) == 1
    assert "Let me look into that for a minute." in json.dumps(encoded)
    assert not with_system_update or "Continue following the rules." in json.dumps(encoded)
    assert messages == original
