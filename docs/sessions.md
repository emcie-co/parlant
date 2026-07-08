# Managing Sessions with the API

Sessions are the conversation state for an agent and a customer. A session owns
its event log, metadata, labels, consumption offsets, and the processing work
that turns customer messages into agent responses.

The session API is centered on two resources:

| Resource | Purpose |
|---|---|
| `/sessions` | Create, list, update, and delete conversation sessions. |
| `/sessions/{session_id}/events` | Append, stream, read, update, and delete events inside a session. |

For authentication and ownership rules, see [`docs/auth.md`](auth.md). The
examples below assume `$PARLANT_URL` points at your server and that any required
`Authorization` header is already set.

## Core Concepts

### Sessions

A session response has this shape:

```json
{
  "id": "sess_123",
  "agent_id": "agent_123",
  "customer_id": "cust_123",
  "creation_utc": "2026-07-06T12:00:00Z",
  "modified_utc": "2026-07-06T12:00:00Z",
  "title": "Checkout support",
  "mode": "auto",
  "consumption_offsets": {"client": 42},
  "metadata": {"priority": "high"},
  "labels": ["vip"]
}
```

Important fields:

| Field | Meaning |
|---|---|
| `agent_id` | The agent that handles the session. |
| `customer_id` | The customer associated with the session. With customer or guest auth, this is derived from the caller rather than trusted from the body. |
| `mode` | `auto` or `manual`. Engines may use `manual` to avoid automatically replying to new customer events. |
| `consumption_offsets.client` | A client-managed bookmark for "events I have consumed". |
| `metadata` | JSON object for application state, routing hints, analytics, or integrations. |
| `labels` | Set-like tags for filtering and lifecycle management. |

### Events

Events are append-only records in a session. They are ordered by `offset` and
grouped by `trace_id`.

```json
{
  "id": "evt_123",
  "source": "customer",
  "kind": "message",
  "offset": 0,
  "creation_utc": "2026-07-06T12:00:00Z",
  "modified_utc": "2026-07-06T12:00:00Z",
  "trace_id": "trace_123",
  "data": {
    "message": "Hi, I need help",
    "participant": {"id": "cust_123", "display_name": "Jane"}
  },
  "metadata": {},
  "deleted": false
}
```

Important fields:

| Field | Meaning |
|---|---|
| `offset` | Monotonic position in the session event log. Use it for incremental reads. |
| `trace_id` | Correlates related events, usually a customer input and the resulting statuses, tools, and agent messages. |
| `kind` | One of `message`, `tool`, `status`, or `custom`. |
| `source` | One of `customer`, `customer_ui`, `human_agent`, `human_agent_on_behalf_of_ai_agent`, `ai_agent`, or `system`. |
| `metadata` | Event-level JSON object. Only metadata is patchable after creation. |
| `deleted` | Indicates an event was deleted. Normal list calls return non-deleted events from the store behavior. |

## Create a Session

```http
POST /sessions?allow_greeting=false
Content-Type: application/json

{
  "agent_id": "agent_123",
  "customer_id": "cust_123",
  "title": "Checkout support",
  "metadata": {"cart_id": "cart_789"},
  "labels": ["checkout", "vip"]
}
```

Parameters:

| Parameter | Where | Effect |
|---|---|---|
| `agent_id` | body | Required. Selects the agent for the session. |
| `customer_id` | body | Optional. Admin/development callers may set it. Customer tokens are scoped to their own customer. Anonymous and guest callers create guest sessions. |
| `title` | body | Optional display title. |
| `metadata` | body | Optional JSON object stored on the session. |
| `labels` | body | Optional labels for filtering and bulk operations. |
| `allow_greeting` | query | If `true`, allows engines that support it to send an initial greeting after creation. Compass currently does not use dynamic greetings here. |

Example:

```bash
curl -X POST "$PARLANT_URL/sessions?allow_greeting=false" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_123",
    "title": "Support chat",
    "metadata": {"source": "web"},
    "labels": ["website"]
  }'
```

If anonymous guest auth is enabled, the response may include
`X-Parlant-Guest-Token`. Store it and send it as `Authorization: Bearer ...` on
later requests.

## Read and List Sessions

Read one session:

```http
GET /sessions/{session_id}
```

List sessions:

```http
GET /sessions?agent_id=agent_123&customer_id=cust_123&labels=vip&limit=25&sort=desc
```

Filters and pagination:

| Query parameter | Effect |
|---|---|
| `agent_id` | Only sessions for this agent. |
| `customer_id` | Only sessions for this customer. Customer-authenticated callers are always scoped to their own customer. |
| `labels` | Repeatable list parameter. Only sessions matching the supplied labels are returned. |
| `min_modified_utc` | Only sessions modified at or after this UTC timestamp. Useful for sync jobs. |
| `limit` | Enables cursor pagination and caps the response to 1-100 items. |
| `cursor` | Cursor from a previous paginated response. |
| `sort` | `asc` or `desc`. |

Response shape depends on `limit`:

```http
GET /sessions
```

returns a plain array:

```json
[
  {"id": "sess_1", "agent_id": "agent_123", "...": "..."}
]
```

while:

```http
GET /sessions?limit=25
```

returns a paginated object:

```json
{
  "items": [{"id": "sess_1", "agent_id": "agent_123"}],
  "total_count": 104,
  "has_more": true,
  "next_cursor": "AAAB..."
}
```

## Update a Session

```http
PATCH /sessions/{session_id}
Content-Type: application/json

{
  "title": "Refund issue",
  "mode": "manual",
  "consumption_offsets": {"client": 18},
  "metadata": {
    "set": {"priority": "high", "handoff_reason": "billing"},
    "unset": ["old_priority"]
  },
  "labels": {
    "upsert": ["billing", "vip"],
    "remove": ["checkout"]
  }
}
```

Patch fields:

| Field | Effect |
|---|---|
| `title` | Replaces the session title. |
| `mode` | Sets `auto` or `manual`. |
| `customer_id` | Reassigns the session customer, subject to authorization. |
| `agent_id` | Reassigns the session agent. |
| `consumption_offsets.client` | Updates the client bookmark. |
| `metadata.set` | Merges keys into session metadata. |
| `metadata.unset` | Removes keys from session metadata if present. |
| `labels.upsert` | Adds labels. Existing labels are left as-is. |
| `labels.remove` | Removes labels if present. |

Common uses:

- Mark a conversation as read by setting `consumption_offsets.client` to the
  highest event offset the UI has rendered.
- Route a conversation to human handling with `mode: "manual"` and labels such
  as `["needs-human"]`.
- Attach application state, such as a CRM ticket id, using metadata.

## Delete Sessions

Delete one session:

```http
DELETE /sessions/{session_id}
```

Bulk delete sessions:

```http
DELETE /sessions?agent_id=agent_123&customer_id=cust_123
```

Bulk deletion accepts `agent_id` and `customer_id` filters. If neither is
provided, all sessions visible to the caller and allowed by authorization are
deleted. Use that endpoint carefully.

## Post Events

All event creation uses:

```http
POST /sessions/{session_id}/events
Content-Type: application/json
```

The body always includes:

| Field | Meaning |
|---|---|
| `kind` | Event type: `message`, `custom`, or `status` for direct posting. |
| `source` | Event source. Supported sources depend on `kind`. |
| `message` | Message text for message events that accept explicit text. |
| `data` | JSON payload for custom and status events. |
| `metadata` | Optional event metadata. |
| `participant` | Optional or required participant details, depending on source. |
| `rules` | Optional utterance instructions for generated AI-agent messages. |
| `status` | Required for status events. |

### Customer Message: Start or Continue a Turn

```http
POST /sessions/{session_id}/events?moderation=auto
Content-Type: application/json

{
  "kind": "message",
  "source": "customer",
  "message": "I need help changing my order"
}
```

Effects:

1. Creates a customer message event.
2. Runs moderation if requested.
3. Starts background processing for the session.
4. Returns the customer event immediately, not the final agent response.

Moderation options:

| Query value | Effect |
|---|---|
| `none` | Do not moderate the message. Default. |
| `auto` | Use the configured moderation service. |
| `paranoid` | Use the configured moderation service plus the jailbreak moderation service. |

Moderation results are stored on `data.flagged` and `data.groups`.

You can override the participant only if authorized:

```json
{
  "kind": "message",
  "source": "customer",
  "message": "Hello",
  "participant": {"id": "cust_123", "display_name": "Jane Customer"}
}
```

### AI-Agent Message: Force an Agent Response

To ask the agent to process the current session state:

```http
POST /sessions/{session_id}/events
Content-Type: application/json

{
  "kind": "message",
  "source": "ai_agent"
}
```

This does not accept `message`, because the agent generates the content. Without
`rules`, this starts processing and returns the first status event for that
processing trace. Continue reading events to receive tool events, status updates,
and the final agent message.

To request a specific kind of generated utterance, provide `rules`:

```json
{
  "kind": "message",
  "source": "ai_agent",
  "rules": [
    {
      "action": "Tell the customer that you are checking the order status.",
      "rationale": "buy_time"
    }
  ]
}
```

`rationale` may be `unspecified`, `buy_time`, or `follow_up`.
With `rules`, the endpoint asks the engine to produce the requested utterance
and returns the generated message event.

### Human-Agent Message

```json
{
  "kind": "message",
  "source": "human_agent",
  "message": "I will take it from here.",
  "participant": {"id": "agent_456", "display_name": "Alex"}
}
```

This creates a human message and does not trigger AI processing. `participant`
with `display_name` is required.

### Human-Agent Message on Behalf of the AI Agent

```json
{
  "kind": "message",
  "source": "human_agent_on_behalf_of_ai_agent",
  "message": "Your refund request has been received."
}
```

This records a message with the session agent as the participant, but with a
source that marks it as human-authored on behalf of the AI.

### Custom Event

```json
{
  "kind": "custom",
  "source": "customer_ui",
  "data": {
    "type": "page_view",
    "url": "/checkout"
  },
  "metadata": {
    "browser_session": "abc"
  }
}
```

Custom events require `data`, accept any JSON-serializable object, and do not
trigger AI processing.

### Status Event

```json
{
  "kind": "status",
  "source": "system",
  "status": "typing",
  "data": {"message": "Typing"}
}
```

Supported statuses:

| Status | Typical use |
|---|---|
| `acknowledged` | The system accepted a turn. |
| `processing` | The agent is working. |
| `typing` | The agent is producing a response. |
| `ready` | The turn is finished. |
| `error` | Processing failed. |
| `cancelled` | Special cancellation request, described below. |

For non-cancellation statuses, `data` must be a JSON object. The event is
created directly and does not trigger agent processing.

## Cancel In-Flight Processing

Cancellation is expressed as a status event request:

```http
POST /sessions/{session_id}/events
Content-Type: application/json

{
  "kind": "status",
  "source": "human_agent",
  "status": "cancelled"
}
```

This endpoint does not simply append the supplied event. Instead, it:

1. Finds the active processing task for the session.
2. Cancels that task.
3. Waits until the engine emits a `cancelled` status event for the active trace.
4. Returns that emitted event.

Outcomes:

| Status code | Meaning |
|---|---|
| `201` | Processing was cancelled and a cancellation event was emitted. |
| `409` | There was no active processing task for the session. |
| `504` | The task was cancelled, but no cancellation event appeared before timeout. |

After cancellation, the engine should also emit a terminal `ready` status for
the same trace so clients can clear "thinking" UI.

## Read Events

List events:

```http
GET /sessions/{session_id}/events?min_offset=0&wait_for_data=0
```

Query parameters:

| Query parameter | Effect |
|---|---|
| `min_offset` | Return events with `offset >= min_offset`. Defaults to `0`. |
| `source` | Filter by one source, such as `customer` or `ai_agent`. |
| `kinds` | Comma-separated event kinds, such as `message,status`. |
| `trace_id` | Return only events belonging to one trace. |
| `wait_for_data` | Long-poll timeout in seconds. Default `60`. |
| `sse` | If `true`, stream matching events as Server-Sent Events. |

### Immediate Read

Use `wait_for_data=0` to return immediately:

```bash
curl "$PARLANT_URL/sessions/$SESSION_ID/events?min_offset=0&wait_for_data=0"
```

### Long Polling

Use `min_offset` plus `wait_for_data` to wait for new events:

```bash
curl "$PARLANT_URL/sessions/$SESSION_ID/events?min_offset=12&wait_for_data=30"
```

Behavior:

- If matching events already exist, returns immediately.
- If no matching events exist, waits up to `wait_for_data` seconds.
- If no event arrives before timeout, returns `504`.

Client loop:

```javascript
let offset = 0;

while (true) {
  const res = await fetch(
    `${PARLANT_URL}/sessions/${sessionId}/events?min_offset=${offset}&wait_for_data=30`,
  );

  if (res.status === 504) continue;
  const events = await res.json();

  for (const event of events) {
    render(event);
    offset = Math.max(offset, event.offset + 1);
  }
}
```

### Server-Sent Events

Use `sse=true` to receive a stream:

```javascript
const source = new EventSource(
  `${PARLANT_URL}/sessions/${sessionId}/events?min_offset=0&sse=true&wait_for_data=60`,
);

source.onmessage = (message) => {
  const event = JSON.parse(message.data);
  render(event);
};
```

The stream sends each matching event as:

```text
data: {"id":"evt_123", ...}
```

and closes when no matching event arrives within `wait_for_data` seconds.
Reconnect with the next `min_offset` to continue.

### Filtering Examples

Only messages:

```http
GET /sessions/{session_id}/events?kinds=message&wait_for_data=0
```

Only status updates for one trace:

```http
GET /sessions/{session_id}/events?kinds=status&trace_id=trace_123&wait_for_data=30
```

Only agent-visible output:

```http
GET /sessions/{session_id}/events?source=ai_agent&kinds=message&wait_for_data=0
```

## Read One Event

```http
GET /sessions/{session_id}/events/{event_id}
```

Optional query parameters:

| Query parameter | Effect |
|---|---|
| `wait_for_completion=true` | Wait until a streaming event completes before returning. |
| `wait_for_data` | Timeout in seconds for completion or streaming updates. |
| `sse=true` | Stream updates to this event until it completes. |

Streaming events use `data.chunks`. Completion is represented by a final
`null` chunk:

```json
{
  "data": {
    "message": "Hello there",
    "chunks": ["Hello", " there", null]
  }
}
```

Wait for the complete event:

```http
GET /sessions/{session_id}/events/{event_id}?wait_for_completion=true&wait_for_data=60
```

Stream updates for a single event:

```javascript
const source = new EventSource(
  `${PARLANT_URL}/sessions/${sessionId}/events/${eventId}?sse=true&wait_for_data=60`,
);

source.onmessage = (message) => {
  const event = JSON.parse(message.data);
  renderStreamingEvent(event);
};
```

For non-streaming events, `sse=true` sends the event once and closes.

## Update Event Metadata

Only event metadata is patchable:

```http
PATCH /sessions/{session_id}/events/{event_id}
Content-Type: application/json

{
  "metadata": {
    "set": {"reviewed": true, "sentiment": "positive"},
    "unset": ["temporary_flag"]
  }
}
```

This is useful for UI flags, review state, moderation annotations, or
integration bookkeeping. It does not change event content.

## Delete Events and Regenerate from an Earlier Point

```http
DELETE /sessions/{session_id}/events?min_offset=12
```

This deletes all events with `offset >= min_offset` and truncates stored agent
state that belongs to the deleted trace range.

Important constraint:

The event at `min_offset` must be the first non-deleted event of its `trace_id`.
If it is not, the API returns `422` with:

```text
Cannot delete events with offset < min_offset unless they are the first event of their trace ID
```

Why this matters:

- A user turn produces several events with the same `trace_id`, such as customer
  message, statuses, tool calls, and agent message.
- Deleting from the middle of that trace would leave an inconsistent turn.
- To regenerate a response, delete from the first event of the trace you want
  to replay.

Typical regenerate flow:

1. Find the customer message you want to replay.
2. Use its `offset` as `min_offset`.
3. Delete events from that offset.
4. Post the replacement customer message.
5. Listen for the new trace's events.

## Recommended Client Patterns

### Simple Chat UI

1. `POST /sessions`.
2. `POST /sessions/{id}/events` with `kind=message`, `source=customer`.
3. Start `GET /sessions/{id}/events?min_offset=<next>&sse=true`.
4. Render status events for "thinking" and message events for visible chat.
5. Store the highest rendered offset with `PATCH /sessions/{id}` and
   `consumption_offsets.client`.

### Long-Polling Chat UI

Use this when SSE is not available:

1. Keep a local `nextOffset`.
2. Call `GET /sessions/{id}/events?min_offset=<nextOffset>&wait_for_data=30`.
3. On `504`, retry.
4. On events, render them and advance `nextOffset`.

### Human Handoff

1. `PATCH /sessions/{id}` with `mode: "manual"` and labels such as
   `["needs-human"]`.
2. Human operator posts messages with `source: "human_agent"`.
3. If the AI is currently processing, post a `cancelled` status event first.
4. When returning to automation, `PATCH /sessions/{id}` with `mode: "auto"`.

### Stop Button

When a response is in flight:

1. POST a cancellation status event.
2. Wait for the returned `cancelled` event.
3. Continue listening for the terminal `ready` status on the same trace.
4. Clear the client-side "thinking" state when `ready` appears.

### Sync Job

To mirror sessions into another system:

1. Call `GET /sessions?limit=100&sort=asc&min_modified_utc=<last_sync_time>`.
2. Follow `next_cursor` until `has_more` is false.
3. For changed sessions, call `GET /sessions/{id}/events?wait_for_data=0`.
4. Store the latest sync timestamp.

## Error Reference

| Status | Common cause |
|---|---|
| `401` | Missing or invalid credentials. |
| `403` | Caller is authenticated but cannot access this session or operation. |
| `404` | Session or event does not exist. |
| `409` | Cancellation requested when no processing task is active. |
| `422` | Invalid body, unsupported event source/kind combination, missing required field, or invalid event deletion offset. |
| `504` | Long-poll timed out, or cancellation did not produce a cancellation event before timeout. |

## Endpoint Summary

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/sessions` | Create a session. |
| `GET` | `/sessions` | List sessions, optionally filtered and paginated. |
| `GET` | `/sessions/{session_id}` | Read one session. |
| `PATCH` | `/sessions/{session_id}` | Update title, mode, ownership fields, metadata, labels, and consumption offset. |
| `DELETE` | `/sessions/{session_id}` | Delete one session. |
| `DELETE` | `/sessions` | Bulk delete sessions by filter. |
| `POST` | `/sessions/{session_id}/events` | Create a message, custom, or status event. Also used to request cancellation. |
| `GET` | `/sessions/{session_id}/events` | List, long-poll, or SSE-stream events. |
| `GET` | `/sessions/{session_id}/events/{event_id}` | Read or stream one event. |
| `PATCH` | `/sessions/{session_id}/events/{event_id}` | Update event metadata. |
| `DELETE` | `/sessions/{session_id}/events` | Delete events from an offset onward. |
