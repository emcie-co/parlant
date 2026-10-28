# AG-UI — connecting Parlant to AG-UI / CopilotKit frontends

## Overview

[AG-UI](https://docs.ag-ui.com) (the Agent-User Interaction Protocol) is an open,
event-based protocol that connects an agent backend to a frontend. It standardizes the
stream a UI needs — streaming chat, tool calls, shared state, and generative UI — so any
AG-UI-compatible frontend (for example [CopilotKit](https://www.copilotkit.ai)'s React
components) can talk to any AG-UI-compatible backend. It is the frontend-facing companion to
MCP (agent-to-tools) and A2A (agent-to-agent).

It works like this: a client sends a **`RunAgentInput`** (a thread id, a run id, the
messages, shared state, and any frontend tools), and the agent replies with a **stream of typed events** over Server-Sent Events (SSE): run lifecycle (`RUN_STARTED`, `RUN_FINISHED`,
`RUN_ERROR`), text (`TEXT_MESSAGE_START` / `_CONTENT` / `_END`), tool calls
(`TOOL_CALL_START` / `_ARGS` / `_END` / `_RESULT`), state (`STATE_SNAPSHOT` / `STATE_DELTA`),
and extension events (`CUSTOM`, `RAW`).

Parlant is already an event-streaming agent backend: a session has an event log, and clients
read it over SSE. So supporting AG-UI does **not** require any change to Parlant — you write a
small **adapter** that re-shapes Parlant's session events into AG-UI events. Parlant keeps
owning the *behavior* (rules, tools, matching, compliance); AG-UI/CopilotKit owns
the *UX* (rendering, generative UI, components).

This document is a guide to building that adapter yourself. The adapter is a thin
translation service that you write and host, so it lives in your codebase, not in
Parlant's.

## Motivation

Teams that build on Parlant want a controlled, compliant agent, but often do not want to
build a chat UI, streaming code, and generative-UI rendering from scratch. AG-UI is the
standard that lets them use an existing frontend (CopilotKit and others) instead.

Pairing the two gives a clean division of labor:

- **Parlant** decides *what the agent does* — which rules match, which tools run, what the
  agent may say — with all its guarantees intact.
- **AG-UI / CopilotKit** decides *how it looks* — streaming bubbles, tool/widget rendering,
  prebuilt components.

Because the bridge is an adapter over Parlant's existing HTTP/SSE API, none of this requires
forking or modifying Parlant.

## Architecture

Run the adapter as its own small HTTP service that sits between the frontend and a normal
Parlant server:

```
CopilotKit / AG-UI client  ──POST /agui (RunAgentInput)──▶  Adapter  ──REST + SSE──▶  Parlant server
                           ◀────── AG-UI event stream ─────           ◀── session events ──
```

The adapter has three parts, built below:

1. **Translator** — a pure function from a Parlant event to a list of AG-UI events.
2. **Source** — drives a Parlant session over its HTTP/SSE API and yields the turn's events.
3. **Server** — a `POST /agui` endpoint that frames the run and streams encoded AG-UI events.

Install the dependencies in the adapter's own environment:

```bash
pip install ag-ui-protocol fastapi uvicorn httpx
```

You also need a running Parlant server with an agent (see the tutorial's last step).

## Step 1 — The translator (Parlant events → AG-UI events)

Parlant's event feed yields one JSON object per event, with `kind`
(`message`/`tool`/`status`/`custom`), `source`, `id`, `trace_id`, and a `data` payload. The
translator maps each to the AG-UI event(s) it represents. It is *stateful within a run*: a
streaming Parlant message arrives as several snapshots whose `chunks` list grows and finally
ends with a `None` terminator, so we emit one `TEXT_MESSAGE_START`, one
`TEXT_MESSAGE_CONTENT` per new chunk, and one `TEXT_MESSAGE_END`.

```python
# translator.py
from __future__ import annotations

import json
from typing import Any

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    RunErrorEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class ParlantToAGUITranslator:
    """Maps one Parlant turn's events into AG-UI events."""

    def __init__(self) -> None:
        self._started: set[str] = set()
        self._emitted: dict[str, int] = {}
        self._ended: set[str] = set()
        self._seen_calls: set[str] = set()

    def translate(self, event: dict[str, Any]) -> list[BaseEvent]:
        kind, source = event.get("kind"), event.get("source")
        if kind == "message" and source == "ai_agent":
            return self._message(event)
        if kind == "tool":
            return self._tool(event)
        if kind == "status":
            return self._status(event)
        if kind == "custom":
            return [CustomEvent(type=EventType.CUSTOM, name="parlant.custom", value=event.get("data"))]
        return []  # customer / UI / human-agent events are not the agent's output

    def _message(self, event: dict[str, Any]) -> list[BaseEvent]:
        out: list[BaseEvent] = []
        mid = str(event["id"])
        data = event.get("data") or {}
        chunks = data.get("chunks")

        if mid not in self._started:
            self._started.add(mid)
            self._emitted[mid] = 0
            out.append(TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=mid, role="assistant"))

        if chunks is not None:  # streaming: list of text pieces, terminated by None
            content = [c for c in chunks if c is not None]
            for delta in content[self._emitted[mid]:]:
                out.append(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=delta))
            self._emitted[mid] = len(content)
            if chunks and chunks[-1] is None and mid not in self._ended:
                self._ended.add(mid)
                out.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid))
        else:  # block mode: whole message at once
            message = data.get("message") or ""
            if self._emitted[mid] == 0 and message:
                out.append(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=message))
                self._emitted[mid] = 1
            if mid not in self._ended:
                self._ended.add(mid)
                out.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid))
        return out

    def _tool(self, event: dict[str, Any]) -> list[BaseEvent]:
        out: list[BaseEvent] = []
        for index, call in enumerate((event.get("data") or {}).get("tool_calls", [])):
            # Parlant tool calls have no per-call id; synthesize a stable one.
            call_id = f"{event['id']}:{index}"
            if call_id in self._seen_calls:
                continue
            self._seen_calls.add(call_id)
            result = call.get("result") or {}

            out.append(ToolCallStartEvent(type=EventType.TOOL_CALL_START, tool_call_id=call_id, tool_call_name=call.get("tool_id", "")))
            out.append(ToolCallArgsEvent(type=EventType.TOOL_CALL_ARGS, tool_call_id=call_id, delta=json.dumps(call.get("arguments", {}), ensure_ascii=False, sort_keys=True)))
            out.append(ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=call_id))
            out.append(ToolCallResultEvent(type=EventType.TOOL_CALL_RESULT, message_id=f"{call_id}:result", tool_call_id=call_id, content=_stringify(result.get("data"))))

            # Frontend-only rich output (files, widgets) rides result.metadata.
            if metadata := result.get("metadata"):
                out.append(CustomEvent(type=EventType.CUSTOM, name="parlant.tool_result_metadata", value={"tool_call_id": call_id, "metadata": metadata}))
        return out

    def _status(self, event: dict[str, Any]) -> list[BaseEvent]:
        data = event.get("data") or {}
        if data.get("status") == "error":
            return [RunErrorEvent(type=EventType.RUN_ERROR, message=_stringify(data.get("data") or data.get("message") or "error"))]
        if data.get("message"):  # agent narration / "thinking"
            return [CustomEvent(type=EventType.CUSTOM, name="parlant.status", value=data)]
        return []
```

Two shape notes this encodes:

- Parlant records a tool call as a **single** event carrying name + arguments + result; AG-UI
  models a call as a **sequence** (`START → ARGS → END → RESULT`), so `_tool` expands one
  Parlant event into that ordered sequence.
- Parlant tool calls have no id of their own, so we synthesize `<event_id>:<index>` and reuse
  it across the call's four events.

## Step 2 — The source (drive a Parlant session)

The source turns a `RunAgentInput` into Parlant API calls: create or continue a session, post
the customer's latest message, and stream that turn's server-produced events. Keep an
in-memory `threadId → session id` map so a thread stays one Parlant session (and keeps its
history) across runs.

```python
# source.py
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from ag_ui.core import RunAgentInput


def latest_user_message(run_input: RunAgentInput) -> str:
    for message in reversed(run_input.messages):
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", None)
            return content if isinstance(content, str) else str(content or "")
    return ""


class ParlantSource:
    def __init__(self, base_url: str, agent_id: str, *, customer_id: str | None = None, turn_timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._customer_id = customer_id
        self._turn_timeout = turn_timeout
        self._sessions: dict[str, str] = {}  # AG-UI threadId -> Parlant session id

    async def run(self, run_input: RunAgentInput) -> AsyncIterator[dict[str, Any]]:
        message = latest_user_message(run_input)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._turn_timeout) as http:
            session_id = self._sessions.get(run_input.thread_id)
            if session_id is None:
                session = (await http.post(
                    "/sessions",
                    params={"allow_greeting": "false"},
                    json={"agent_id": self._agent_id, "customer_id": self._customer_id},
                )).raise_for_status().json()
                session_id = session["id"]
                self._sessions[run_input.thread_id] = session_id

            posted = (await http.post(
                f"/sessions/{session_id}/events",
                json={"kind": "message", "source": "customer", "message": message},
            )).raise_for_status().json()
            min_offset = posted["offset"] + 1

            async with http.stream(
                "GET",
                f"/sessions/{session_id}/events",
                params={"min_offset": min_offset, "sse": "true", "wait_for_data": int(self._turn_timeout)},
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[len("data:"):].strip())
                    yield event
                    # Stop when the agent's message for this turn is complete.
                    data = event.get("data") or {}
                    if event.get("kind") == "message" and event.get("source") == "ai_agent":
                        chunks = data.get("chunks")
                        if chunks is None or (chunks and chunks[-1] is None):
                            return
                    if event.get("kind") == "status" and data.get("status") in ("ready", "error"):
                        return
```

## Step 3 — The server (`POST /agui`)

The server parses the `RunAgentInput`, frames the run (`RUN_STARTED` … `RUN_FINISHED`, or
`RUN_ERROR` on failure), and streams every translated event as SSE. The AG-UI `EventEncoder`
turns a typed event into a `data: {...}\n\n` line with the protocol's camelCase JSON.

```python
# server.py
from __future__ import annotations

import os
from typing import AsyncIterator

from ag_ui.core import (
    EventType, RunAgentInput, RunErrorEvent, RunFinishedEvent, RunStartedEvent, StateSnapshotEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from source import ParlantSource
from translator import ParlantToAGUITranslator

app = FastAPI(title="Parlant AG-UI adapter")

source = ParlantSource(
    base_url=os.environ.get("PARLANT_URL", "http://localhost:8800"),
    agent_id=os.environ["PARLANT_AGENT_ID"],
)


@app.post("/agui")
async def run_agent(request: Request) -> StreamingResponse:
    run_input = RunAgentInput.model_validate(await request.json())

    async def stream() -> AsyncIterator[str]:
        encoder = EventEncoder()
        translator = ParlantToAGUITranslator()

        yield encoder.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=run_input.thread_id, run_id=run_input.run_id))
        if run_input.state not in (None, {}):
            yield encoder.encode(StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=run_input.state))

        errored = False
        try:
            async for parlant_event in source.run(run_input):
                for ag_event in translator.translate(parlant_event):
                    yield encoder.encode(ag_event)
                    if isinstance(ag_event, RunErrorEvent):
                        errored = True
                if errored:
                    break
        except Exception as exc:  # surface any failure as a run error
            yield encoder.encode(RunErrorEvent(type=EventType.RUN_ERROR, message=str(exc)))
            errored = True

        if not errored:
            yield encoder.encode(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=run_input.thread_id, run_id=run_input.run_id))

    return StreamingResponse(stream(), media_type="text/event-stream")
```

Run it against a Parlant server:

```bash
PARLANT_URL=http://localhost:8800 PARLANT_AGENT_ID=<agent_id> uvicorn server:app --port 9000
```

## Step 4 — Expose a tool that renders a widget

Widgets — and files, images, or any structured UI — travel on a tool's **frontend-only**
`metadata` (see [Rich Content](rich-content.md)). The agent never sees `metadata`; it exists for the
UI. In your Parlant app, define the tool and attach it with an observation:

```python
import parlant.sdk as p


@p.tool
async def show_weather(city: str) -> p.ToolResult:
    return p.ToolResult(
        data=f"Shared a weather card for {city}.",       # what the agent reads / says
        metadata={"widget": {"type": "weather_card", "city": city,
                             "temp_c": 22, "condition": "Sunny"}},  # what the UI renders
    )


async def main() -> None:
    async with p.Server(port=8800) as server:
        agent = await server.create_agent(
            id="my-agent",
            name="Assistant", instructions="You are a helpful assistant.",
        )
        await agent.create_observation(
            condition="the customer asks about the weather in a specific place",
            tools=[show_weather],
        )
    # The server serves after the `async with` block exits; note the agent id it prints
    # and pass it to the adapter as PARLANT_AGENT_ID.
```

When the observation matches, `show_weather` runs, and its `metadata` reaches the frontend as
an AG-UI `CUSTOM` event named `parlant.tool_result_metadata`.

## Step 5 — Consume it

A client POSTs a `RunAgentInput` and reads the SSE stream. Asking "What's the weather in
Tokyo?" yields roughly:

```jsonc
{"type":"RUN_STARTED","threadId":"demo","runId":"r1"}
{"type":"TOOL_CALL_START","toolCallId":"evt_1:0","toolCallName":"show_weather"}
{"type":"TOOL_CALL_ARGS","toolCallId":"evt_1:0","delta":"{\"city\":\"Tokyo\"}"}
{"type":"TOOL_CALL_END","toolCallId":"evt_1:0"}
{"type":"TOOL_CALL_RESULT","toolCallId":"evt_1:0","content":"Shared a weather card for Tokyo."}
{"type":"CUSTOM","name":"parlant.tool_result_metadata",
 "value":{"tool_call_id":"evt_1:0","metadata":{"widget":{"type":"weather_card","city":"Tokyo","temp_c":22,"condition":"Sunny"}}}}
{"type":"TEXT_MESSAGE_START","messageId":"...","role":"assistant"}
{"type":"TEXT_MESSAGE_CONTENT","messageId":"...","delta":"It's sunny and 22°C in Tokyo."}
{"type":"TEXT_MESSAGE_END","messageId":"..."}
{"type":"RUN_FINISHED","threadId":"demo","runId":"r1"}
```

A CopilotKit frontend consumes this stream through its AG-UI client. For a minimal custom
page, read the stream and render text bubbles plus widgets from the `CUSTOM` event:

```javascript
const resp = await fetch("/agui", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    threadId, runId: "r1", state: {},
    messages: [{ id: "u1", role: "user", content }],
    tools: [], context: [], forwardedProps: {},
  }),
});
const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buf = "", bubble = null;
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });
  let i;
  while ((i = buf.indexOf("\n\n")) >= 0) {
    const block = buf.slice(0, i); buf = buf.slice(i + 2);
    const dl = block.split("\n").find((l) => l.startsWith("data:"));
    if (!dl) continue;
    const ev = JSON.parse(dl.slice(5).trim());
    if (ev.type === "TEXT_MESSAGE_START") bubble = newBubble();
    else if (ev.type === "TEXT_MESSAGE_CONTENT") bubble.append(ev.delta);
    else if (ev.type === "CUSTOM" && ev.name === "parlant.tool_result_metadata")
      renderWidget(ev.value?.metadata?.widget);   // e.g. draw the weather card
    else if (ev.type === "RUN_ERROR") showError(ev.message);
  }
}
```

## Mapping reference

| Parlant | AG-UI |
|---|---|
| a turn on a session | a run (`RUN_STARTED` … `RUN_FINISHED`) |
| session id / per-turn `trace_id` | `threadId` / `runId` |
| agent `message` event (streaming `chunks`) | `TEXT_MESSAGE_START` → `TEXT_MESSAGE_CONTENT`* → `TEXT_MESSAGE_END` |
| `tool` event (`tool_calls[]`) | `TOOL_CALL_START` → `TOOL_CALL_ARGS` → `TOOL_CALL_END` → `TOOL_CALL_RESULT` |
| tool `result.metadata` (files, widgets) | `CUSTOM` (`name="parlant.tool_result_metadata"`) |
| `status` = error | `RUN_ERROR` |
| `status` with narration | `CUSTOM` (`name="parlant.status"`) |
| `custom` event | `CUSTOM` (`name="parlant.custom"`) |
| inbound `RunAgentInput.state` | `STATE_SNAPSHOT` (echoed at run start) |

## Unsupported AG-UI features

The core conversational loop — streaming chat, server-side tool calls, generative UI via tool
`metadata`, custom events, and run lifecycle — is fully covered. The following are **not**
covered by the adapter above. Each one can be added with more integration work:

- **Bidirectional shared state (`STATE_DELTA` / JSON Patch).** Inbound `state` is echoed once
  as a `STATE_SNAPSHOT`; there is no live delta sync between the UI's store and the agent. A
  fuller version maps AG-UI state onto Parlant session variables and emits `STATE_DELTA`s.
- **Frontend-defined tools.** `RunAgentInput.tools` (tools the *browser* executes) are ignored
  — Parlant tools run server-side. Honoring them needs a call → pause → client-executes →
  result round-trip that Parlant does not natively provide.
- **Human-in-the-loop `INTERRUPT`.** Pausing a run for user approval/edit relies on the same
  client round-trip gap as frontend tools.
- **Streamed tool-call arguments.** `TOOL_CALL_ARGS` is sent whole, because Parlant records a
  tool call with its arguments already complete.
- **Dedicated thinking/reasoning events.** Parlant narration maps to a `CUSTOM` event
  (`parlant.status`), not AG-UI's `THINKING_*` / `REASONING_*` families.
- **Live per-token text streaming.** The source above reads messages at event granularity;
  true per-token streaming means also reading Parlant's single-event SSE endpoint
  (`GET /sessions/{id}/events/{event_id}?sse=true`). The translator already handles progressive
  chunks, so this is a source-side enhancement, not a protocol gap.
