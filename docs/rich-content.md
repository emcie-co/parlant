# Rich content — files, images, and tool output

## Overview

Conversations are not only text. A customer may send a photo of a broken
product. A tool may open a claim and return a link to a receipt. Parlant
supports this with one simple idea: **it carries references and small pieces of data, never your files.** A file stays in *your* storage. Your frontend
uploads it wherever it already keeps files and gets a URL back. Parlant only
passes a small, typed reference to that URL through the session.

This is deliberate. Running file storage means upload endpoints, size
limits, access control, and cleanup — work you have almost certainly done
already, under rules Parlant cannot know. So Parlant stores references, and
the files stay where they are. When a tool needs the file, it uses the URL,
and often it just passes the URL on (say, to a vision model) without
downloading anything.

Rich content moves in two directions, and the two sides use different
mechanisms. Keep the terms apart:

- **Inbound** (customer → agent) uses **attachments**: file references on a
  customer message. For example, a customer sends a photo of a damaged
  product to open a warranty claim, or uploads a receipt so a tool can find
  the order.
- **Outbound** (agent → customer) is **not** an attachment. A tool returns
  plain JSON on its result's `metadata` field, and your frontend renders
  it: a download link, a product image, an order summary. There is no way
  to "attach a file" to an agent message, and none is needed — the tool's
  metadata travels on the same turn as the agent's reply.

## Inbound: attachments

### The attachment reference

One small type carries a file reference:

```python
class AttachmentRef(TypedDict):
    url: str                        # where the file lives, in YOUR storage
    media_type: NotRequired[str]    # IANA MIME type: image/jpeg, application/pdf, ...
    caption: NotRequired[str]       # a short description, readable by people and the model
    filename: NotRequired[str]      # original filename, for display and downloads
    metadata: NotRequired[Mapping[str, JSONSerializable]]  # anything else you want to carry
```

`url` is the only required field, and the only locator. It can be a public
URL or a signed URL. Parlant does not open the URL itself.

**What the model sees.** For each attachment, Parlant appends a short note
to the customer's message text. The note uses the caption when present,
then the filename, and marks the kind from `media_type`:

- `[image: photo of a leaking sink pipe]` — caption present, an `image/*` type
- `[file: receipt.pdf]` — no caption, so the filename is used; not an image
- `[image attached]` / `[file attached]` — neither caption nor filename

So the model can read the caption, the filename, and whether the file is an
image. It never sees `url` or `metadata`; those are for your tools and
frontend only. Write captions with this in mind: a good caption is what
lets a rule like *"when the customer sends a photo of a problem, open a repair ticket"* match.

### Attaching a file to a message

Your frontend first uploads the file to your storage. Then it adds the
reference when it posts the customer's message. The `attachments` field
belongs to the events endpoint (see [Sessions](sessions.md) for the full API):

```http
POST /sessions/{session_id}/events
Content-Type: application/json

{
  "kind": "message",
  "source": "customer",
  "message": "Here's the leak under my sink.",
  "attachments": [
    {
      "url": "https://cdn.example.com/u/9f3a.jpg",
      "media_type": "image/jpeg",
      "caption": "photo of a leaking sink pipe"
    }
  ]
}
```

A message may also carry attachments with no text at all. A customer just
sending a photo is a valid message.

**Permission.** Posting attachments needs the
`create_customer_event_with_attachments` operation; without it the request
fails with `403`. The default development policy grants it to customer and
guest callers. A custom authorization policy must grant it explicitly (see
[Authentication](auth.md)).

**Moderation.** Parlant's moderation checks the message *text* only.
Captions and filenames are not moderated, yet they reach the model — treat
them as untrusted input if customers can write them freely. When a message
is flagged, the model sees only the censor notice, and the attachment notes
are dropped along with the text. Tools can still read the attachments.

### Reading attachments in a tool

A tool reads the attachments from the latest customer message through
`p.Session.current`:

```python
@p.tool
async def open_repair_ticket() -> p.ToolResult:
    last = p.Session.current.interaction.last_customer_message
    attachments = last.attachments if last else []
    images = [a for a in attachments if a.get("media_type", "").startswith("image/")]

    if not images:
        return p.ToolResult(
            data="No photo found on the customer's message. Ask them to attach a photo of the problem."
        )

    # The common case: forward the URLs. No file is downloaded here.
    ticket = await crm.create_ticket(  # your own CRM client
        customer_id=p.Customer.current.id,
        photos=[a["url"] for a in images],
    )
    return p.ToolResult(data=f"Opened repair ticket {ticket.id}.")
```

Note the no-photo path: it returns a `ToolResult` that tells the model what
to do next, so the agent can ask the customer for a photo. Raising an
exception instead would only give the model an error to improvise around.

`.attachments` is the list of `AttachmentRef`s on that message. A tool can
download the file from its URL if it really needs to. Often it does not.

## Outbound: tool results for the frontend

A tool result has two key fields:

- `data` is what the agent reads and reasons with.
- `metadata` is only for the frontend. The agent never sees it.

You can put anything the frontend needs into `metadata`: a link, a file
reference, or a small JSON object your UI can render. Parlant does not
define a widget format; your frontend decides how to display the data.

```python
@p.tool
async def generate_invoice(order_id: str) -> p.ToolResult:
    url = await billing.render_invoice_pdf(order_id)   # stored in YOUR storage

    return p.ToolResult(
        data=f"Invoice for order {order_id} is ready and shown to the customer for download.",
        metadata={"invoice": {"url": url, "media_type": "application/pdf"}},
    )
```

The model reads `data`, so it can tell the customer the invoice is ready.
It never sees the URL. The URL lives in `metadata`, which only your
frontend reads — say, to render a download card.

### How the frontend gets it

You do not need a new endpoint or an extra request. Tool events arrive in
the feed the frontend already reads: the `list_events` long-poll and the
SSE stream deliver `kind=tool` events alongside `message` and `status`
events (see [Sessions](sessions.md)).

Every event created during a single turn shares the same `trace_id`. To
show tool output next to the agent's message, take the `trace_id` from the
message event you rendered, and match it against the tool events from the
same stream.

A tool event's `data` holds the turn's tool calls. Trimmed to the fields
that matter here:

```jsonc
{
  "kind": "tool", "source": "system", "trace_id": "trc_9f…",
  "data": {
    "tool_calls": [
      {
        "tool_id": "built-in:generate_invoice",
        "arguments": { "order_id": "5567" },
        "result": {
          "metadata": { "invoice": { "url": "…", "media_type": "application/pdf" } },
          "data": "Invoice for order 5567 is ready…"
        }
      }
    ]
  }
}
```

The frontend reads this path: **`toolEvent.data.tool_calls[i].result.metadata`**.
Three things to get right:

- **`tool_calls` is a list with one entry per call, not per tool.** If a
  tool runs three times in one turn, it appears three times. Loop over the
  list, and tell entries apart by `arguments`.
- **`tool_id` is qualified** as `service:tool`. SDK tools appear as
  `built-in:generate_invoice`, not `generate_invoice` — match accordingly.
- **Skip the `__parlant__` key** if it appears in `result.metadata`. It is
  Parlant's own namespace (error details and similar), not your tool's data.

### Fetching a turn's tool calls by trace id

If you show the live event stream, you already receive these tool events.
Sometimes you want to load one turn's tool calls on their own — for
example, when loading history. There is no special endpoint. Ask the events
endpoint for `kind=tool` events with that `trace_id`, and pass
`wait_for_data=0` so it returns at once (an empty list for a turn with no
tools):

**Python** (with `httpx` — any HTTP client works):

```python
import httpx

PARLANT_URL = "http://localhost:8800"


def tool_calls_for_trace(session_id: str, trace_id: str) -> list[dict]:
    events = httpx.get(
        f"{PARLANT_URL}/sessions/{session_id}/events",
        params={
            "trace_id": trace_id,   # from the message event you rendered
            "kinds": "tool",
            "wait_for_data": 0,     # return what's there; do not long-poll
        },
    ).raise_for_status().json()

    return [call for e in events for call in e["data"]["tool_calls"]]


for call in tool_calls_for_trace(session_id, trace_id):
    metadata = call["result"]["metadata"]
    ...  # render whatever your tools put here
```

**TypeScript** (plain `fetch`):

```typescript
const PARLANT_URL = "http://localhost:8800";

async function toolCallsForTrace(sessionId: string, traceId: string) {
  const params = new URLSearchParams({
    trace_id: traceId,          // from the message event you rendered
    kinds: "tool",
    wait_for_data: "0",         // return what's there; do not long-poll
  });
  const res = await fetch(`${PARLANT_URL}/sessions/${sessionId}/events?${params}`);
  const events = await res.json();
  return events.flatMap((e: any) => e.data.tool_calls ?? []);
}

for (const call of await toolCallsForTrace(sessionId, traceId)) {
  const metadata = call.result.metadata ?? {};
  // render whatever your tools put here
}
```

Each call includes its `result.metadata`. It also includes `result.data`
when the caller has `READ_TOOL_RESULT_DATA`; otherwise that field is
replaced with the marker `{"redacted": true}`. The shape is kept, so check
for the marker rather than for a missing field.

## Notes and limits

- **URLs are yours to protect.** Parlant treats `url` as plain text and
  never opens it. If a file must stay private, give Parlant a signed URL
  that expires.
- **Parlant does not validate attachments.** No count limit, no size check,
  no URL-scheme check — it stores what your frontend sends. Validate at
  your edge: cap the number of attachments, check types, and only accept
  URLs from your own storage.
- **When a tool downloads a URL, it is your network call.** Treat it like
  any other outbound request: allow only known hosts, and set timeouts. A
  URL from an attacker can be used for server-side request forgery (SSRF).
- **You provide `media_type`; Parlant does not check it.** Parlant never
  sees the file, so it cannot confirm the type. Check it yourself if it
  matters.
- **Tool `metadata` is hidden from the agent, and it is not moderated.** It
  does not affect the agent's thinking, and authorization and moderation do
  not check it. Do not put agent-relevant facts there; use `data` for
  those.
- **Rendering is the frontend's job.** The server's built-in chat page
  shows only text and status. The embeddable chat widget (`widget/` in the
  repo) does render tool `result.metadata` and supports attaching files
  from the composer — use it if you don't want to build a consumer
  yourself. It has one built-in widget, `choices`: a tool returns
  `{"widget": {"type": "choices", "options": [{"id": …, "label": …}]}}`
  and the visitor's click is sent as their own message (see the widget
  README's *Rich content* section).
- **For frontend-only data outside a tool call**, post a `kind=custom`
  event on the session — same feed, no tool needed (see
  [Sessions](sessions.md)).
- **Session memory and attachments are separate.** Attachments belong to
  single messages within one session. They are not long-term customer
  memory (see [Memory](memory.md)).
