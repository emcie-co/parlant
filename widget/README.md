# Parlant Chat Widget

The official React widget for customer-facing Parlant conversations — a floating chat
launcher and panel you can drop onto any website.

## See it running first

The widget ships inside the `parlant` pip package, so if you have Parlant installed you can
run the whole demo with one command:

```sh
export OPENAI_API_KEY=sk-…       # the sample agent uses OpenAI for chat and vision
parlant widget playground
```

That copies the playground into your OS cache directory (it prints the path), installs its
dependencies, starts the **bundled sample agent**, and opens the playground at
http://localhost:5173 — a live page
where you toggle every prop (display mode, theme, streaming style, sounds, …) against a real
agent and copy the resulting config. Press Ctrl+C to stop both.

Useful flags: `--signed-in` runs the agent with customer-JWT auth and prints a token to paste
in; `--no-agent` skips the sample agent so you can point the playground at your own server;
`--port` moves the dev server; `--dir` copies the playground somewhere you'd rather hack on it.

The cache directory is deliberate: the playground's real cost is its npm install, so keeping
it there means one install per machine rather than one per project, and nothing is left in the
directory you ran from.

Working inside a Parlant checkout instead? The playground runs straight from this folder:

```sh
cd widget
npm install
npm run dev        # opens the playground at http://localhost:5173
```

Once you've played with it, here's how to put it in your own app.

## Distribution

It is distributed **shadcn-style**: this is not an npm package you import, it's a reference
implementation you **copy into your app and own**. Copy the single `parlant-chat/` folder,
restyle it by redefining a few CSS variables, swap the transport, add your own tool widgets —
it's your code from the moment you copy it. **No Tailwind, no build config** — the widget
ships self-contained CSS.

## Why copy-paste instead of a package

A customer chat widget lives in *your* product's design system, auth, and storage. A locked
npm component fights all three. Owning the source means:

- Theme it by overriding a handful of CSS variables; no Tailwind, no fighting a library's styles.
- Wire it to your auth and your file storage (Parlant is bring-your-own storage).
- Render your own rich content (tool result widgets, attachments) however you like.
- No version churn — you upgrade by re-copying the parts you want.

## Requirements

- React 18+. **No Tailwind, no CSS framework, no build config** — the widget ships its own
  self-contained stylesheet.
- A running Parlant server reachable from the browser, and an agent id.
- No runtime dependency on a Parlant client SDK — the widget talks to the server through a
  small vendored transport (`parlant-chat/lib/transport.ts`) you can read in one sitting.
- No font setup — the widget ships the **Inter** font inline in `styles.css` (imported by
  `chat-widget.tsx`), scoped to the widget so it never touches your page's typography. Override
  it with the `fontFamily` prop (any family or full stack).

## Install & integrate

Two steps: **copy one folder → render it.** It works with zero styling config; theming is
optional (step 3).

### 1. Copy one folder

The entire widget is the single `parlant-chat/` directory — everything else in `widget/` is
the dev/demo harness. From your app's directory:

```sh
parlant widget init                  # copies it to ./parlant-chat
parlant widget init src/parlant-chat # or wherever you keep vendored code
```

This is a plain file copy, so it needs no node toolchain and makes no network call — the
source ships inside the pip package. It writes a `.parlant-widget-version` file recording the
Parlant version the copy came from, since the widget's transport speaks that version's HTTP
API. Re-run it with `--force` after a major server upgrade.

From a checkout, copying the folder by hand does the same thing:

```sh
cp -r parlant-chat ~/your-app/src/parlant-chat
```

It's self-contained — transport, hooks, components, and **all styling** (bundled Inter font,
reveal animations, theme tokens, and every component's CSS) live inside `parlant-chat/`:

```
parlant-chat/
  index.ts       the public API (ChatWidget, registerWidget, types)
  lib/           transport (fetch + SSE), events, markdown renderer, cn
  hooks/         session, streaming, /healthz, scroll-to-bottom
  components/    the UI — ChatWidget plus its parts
  styles.css     Inter font + theme tokens + reveal animations + all component styles
```

### 2. Render it

Import from the folder's barrel. `styles.css` is imported by the widget itself, so your
bundler (Vite, Next, …) just needs to allow CSS imports — no separate `import "./styles.css"`.

```tsx
import { ChatWidget } from "@/parlant-chat";

export function App() {
  return <ChatWidget serverUrl="https://parlant.your-app.com" agentId="your-agent-id" />;
}
```

It renders fully styled out of the box (light + dark, following the OS), with the Inter font
scoped to the widget so it never touches your page's typography.

### 3. Theme it (optional)

The widget defines its palette as CSS variables on `.parlant-widget`, with light + dark
defaults. To match your brand, redefine any of them (they're shadcn-style HSL triplets used as
`hsl(var(--token))`):

```css
.parlant-widget {
  --background: 0 0% 100%;
  --foreground: 240 10% 4%;
  --primary: 240 6% 10%;
  --primary-foreground: 0 0% 98%;
  --muted: 240 5% 96%;
  --muted-foreground: 240 4% 46%;
  --border: 240 6% 90%;
  --ring: 240 5% 65%;
  --destructive: 0 72% 51%;
}
```

**Already use shadcn/ui?** Set these to your own `--background`/`--foreground`/… values (or map
them) and the widget inherits your theme. For dark mode, add a `.dark` ancestor or
`data-theme="dark"` on/above `.parlant-widget` (it also follows `prefers-color-scheme` by
default).

That's the batteries-included default: a launcher bubble that opens a streaming chat panel.
Wire `onUpload` for attachments and `auth` for signed-in customers (below); everything
under `ChatWidget` is composable if you want to build your own surface.

## Display modes

`mode` picks where the widget sits. All four render the same chat surface — only the
chrome around it differs.

```tsx
// input (default) — a composer docked in the bottom-right corner
<ChatWidget serverUrl="…" agentId="…" />

// bar — a message bar centered at the bottom of the screen
<ChatWidget serverUrl="…" agentId="…" mode="bar" />

// floating — a launcher bubble that opens a panel
<ChatWidget serverUrl="…" agentId="…" mode="floating" />

// inline — fills its container
<div className="h-[600px] w-[380px]">
  <ChatWidget serverUrl="…" agentId="…" mode="inline" />
</div>
```

- **`input`** (default) — a floating composer in the corner. It starts collapsed, shows your
  `openingSuggestions` as starter chips, and grows upward into the full panel on the first message.
  The session is created lazily, on that first send.
- **`bar`** — the same surface, centered at the bottom of the screen and wider, for pages
  where the chat is the main action rather than a corner afterthought. It grows into the
  panel from the center.
- **`floating`** — a launcher bubble; the panel opens from the corner. Your `openingSuggestions`
  show as starter chips above the input while the opened conversation is still empty.
- **`inline`** — no chrome at all; it fills whatever container you give it. It shows
  `openingSuggestions` the same way `floating` does.

On phones, `input` and `bar` both dock full-width at the bottom and open full-screen.

## Opening as a sidebar

`mode` controls the launcher or dock; `openMode` controls the opened chat. The
usual panel remains the default (`openMode="expanded"`). Use `openMode="sidebar"`
for a full-height right column that reserves space beside the page. Expand widens
that column; minimize releases it. On mobile it fills the screen, without reserving
page space or showing the size toggle. Inline mode continues to fill its container.

```tsx
<ChatWidget serverUrl="…" agentId="…" openMode="sidebar" />
```

The host page keeps its own styling. Layouts that depend on available content
width can use container queries to respond to the column beside the sidebar.

## Suggestion chips

Use `openingSuggestions` for conversation starters and `persistentSuggestions`
for prompts that remain available after the conversation begins. Selecting a
persistent chip sends its text and removes it for that session, including when
this browser resumes it. A new session gets the prompts again. Typing the same
text manually does not select the chip.

`maxVisibleSuggestions` defaults to 3. Opening suggestions take priority, then
unchosen persistent suggestions fill the remaining slots, in their configured
order. Duplicate text appears only once. Set the limit to 0 to hide chips.

```tsx
<ChatWidget
  serverUrl="…"
  agentId="…"
  openingSuggestions={["What can you do?"]}
  persistentSuggestions={["How do I get started?", "Talk to someone"]}
  maxVisibleSuggestions={3}
/>
```

## The panel

- **Header** — shows the agent's name and an avatar with 10px corners with its uppercase initials, both
  fetched from `GET /agents/{id}`. Pass `title` to override the name.
- **Expand / minimize** — the header has an expand toggle (roughly doubles the panel width for
  longer reading) and a minimize control. Minimize returns to the launcher; the widget is never
  truly "closed" (there's no destructive close on a support chat).
- **Preface** — pass `preface` to show a muted intro line above the conversation at the start of
  a session (e.g. "You're chatting with our support assistant.").
- **New chat** — pass `showNewChat` to add a header button (with a hover tooltip) that asks for
  confirmation, then starts a fresh session: it clears the stored session id, bootstraps a new one,
  and resets the scroll — the old thread is left intact server-side.

## Authentication

The widget supports two flows — pick based on whether your visitors are anonymous or
signed-in. **Never put an admin/API key in browser code**: the widget only ever holds a guest
or per-customer token.

### Anonymous visitors — guest sessions (default)

With no auth configured, the widget uses **guest sessions**. `POST /sessions` returns an
`X-Parlant-Guest-Token`, which the widget stores and sends as `Authorization: Bearer …` on
every later request. Zero config beyond the server URL and agent — good for public sites and
docs. Your server's authorization policy must allow guests; see
[`docs/auth.md`](../docs/auth.md).

```tsx
// example/guest.tsx
<ChatWidget serverUrl="https://parlant.your-app.com" agentId="your-agent-id" />
```

The guest token and session ids are kept in `localStorage`, so a returning visitor resumes
their thread. The server renews the token while the visitor keeps coming back (up to its
maximum guest identity age, 28 days by default), and the widget stores each renewal.

### Signed-in customers — your token

When you already know who the user is, bind the session to the real customer. Mint a Parlant
**customer token on your backend** (where your admin credentials live) and hand the widget a
way to fetch it. The widget sends it as `Authorization: Bearer …`, and Parlant derives the
`customer_id` from the token. `identityKey` is your stable, app-owned user key; the widget uses
it only to keep different customers' locally persisted sessions separate. It is never sent to
Parlant and is not a substitute for the token's `sub` claim.

```tsx
// example/authenticated.tsx
<ChatWidget
  serverUrl="https://parlant.your-app.com"
  agentId="your-agent-id"
  auth={{
    type: "customer",
    identityKey: currentUser.id,
    // Returns a Parlant customer token minted by YOUR backend. The widget
    // calls this initially and again when an authenticated request gets a 401.
    getToken: async () => {
      const res = await fetch("/api/parlant-token", { credentials: "include" });
      return (await res.json()).token;
    },
  }}
/>
```

Your `/api/parlant-token` endpoint authenticates the user with your own session/cookies and
returns a short-lived Parlant customer token. The browser never sees your admin key. Customer
tokens are cached only in memory; the widget never writes them to browser storage. In customer
mode the guest flow is skipped entirely. If any authenticated operation gets a 401, the widget
discards its cached customer token, calls `auth.getToken` again, and retries that request once.

#### Backend setup for real sign-in

Parlant is **not** an identity provider — it only *verifies* a JWT your backend mints. To make
the signed-in flow bind sessions to a real customer (see [`docs/auth.md`](../docs/auth.md) for the
full model):

1. **Run the server with auth enabled** — an `AuthConfig` (or `PARLANT_*` env vars) with an
   `admin_api_key` and a `customer_jwt_secret` (the HS256 secret you'll sign tokens with), plus a
   `guest_token_secret`.
2. **Make the customer known** — `await server.create_customer(id="user_1234", …)`, or set
   `auto_provision_customers=True` so the first valid JWT for an unknown `sub` creates it.
3. **Mint a JWT** signed with that `customer_jwt_secret`, `sub` = the Parlant `CustomerId`, short
   `exp`. Your `/api/parlant-token`-style endpoint returns it; Parlant derives the customer from
   the token (never send `customer_id` in the body).

> **Dev-mode caveat:** a server started with no auth (the default / not `PARLANT_ENV=production`)
> runs a development policy that **allows everything and ignores the token entirely**. The
> signed-in example will *appear* to work but the session isn't bound to a customer until you do
> the three steps above.

To try the signed-in flow for real, the sample agent has an auth-enabled variant:

```bash
# 1. Run Agent Tom with customer-JWT verification + auto-provisioning (port 8811)
OPENAI_API_KEY=sk-... uv run widget/sample-agent/signed_in_agent.py

# 2. Mint a customer token (plays the role of your backend; expires after an hour)
uv run widget/sample-agent/mint_token.py                      # Dana Cohen / crm-demo-1
uv run widget/sample-agent/mint_token.py user-42 "Avi Levi"   # your own id + name
```

In the playground, pick **Signed in**, set the server URL to `http://localhost:8811`, and paste
the minted JWT into the **Customer token** field — its `sub` claim doubles as the identity key
there. The playground is intentionally a manual pasted-token harness; in your own app,
`auth.getToken` fetches the JWT from your backend and you pass `identityKey: user.id` yourself.

Both flows ship as runnable pages in `example/` — `guest.tsx` and `authenticated.tsx`.

## How it talks to Parlant

The transport (`lib/transport.ts`) is a thin wrapper over Parlant's REST + SSE API. It persists
one session per server, agent, and auth identity so a returning visitor keeps the right thread:

1. **Create/continue a session** — `POST /sessions?allow_greeting=false` with `{ agent_id }`.
   If guest auth is enabled the response carries an `X-Parlant-Guest-Token`; the widget stores
   it and sends it as `Authorization: Bearer …` afterwards. The session id is cached in
   `localStorage`.
2. **Send a message** — `POST /sessions/{id}/events` with
   `{ kind: "message", source: "customer", message, attachments? }`.
3. **Stream events** — `GET /sessions/{id}/events?sse=true&min_offset=…&wait_for_data=60`,
   reconnecting from the last offset on drop. Each line is an `EventDTO`. For token-by-token
   streaming, a message event's growing text is read from
   `GET /sessions/{id}/events/{event_id}?sse=true` (chunks until a `null` terminator).

The event kinds the widget consumes:

| Kind | What the widget does |
|---|---|
| `message` (source `ai_agent`) | Render an agent bubble; stream its `chunks` in. |
| `message` (source `customer`) | Render the customer's own bubble. |
| `status` | Drive the thinking line; `message` (the agent's narration) is its text; `error` shows an error. |
| `tool` | Render `result.metadata` as a widget (see [Rich content](#rich-content)). |
| `custom` | App-defined; ignored by default. |

## Architecture

```
parlant-chat/
  index.ts              # public API barrel (ChatWidget, registerWidget, types)
  styles.css            # Inter font (@font-face) + theme tokens + reset + reveal animations
                        #   + Markdown styles + every component's CSS (self-contained, no Tailwind)
  lib/
    transport.ts        # fetch + SSE: session, send, stream, /healthz, agent profile
    guest-identity.ts   # one guest identity per server: token + each agent's conversation, shared by tabs
    events.ts           # normalized event/message types
    links.ts            # UTM tagging for rendered links (appendUtm)
    markdown.ts         # tiny zero-dep Markdown parser (+ streaming-safe completion)
    cn.ts               # className helper
  hooks/
    use-parlant-session.ts   # bootstrap/continue a session (guest token + localStorage)
    use-parlant-stream.ts    # SSE -> ordered messages + streaming text + reasoning + phase
    use-server-health.ts     # /healthz polling -> online flag (offline banner)
    use-stick-to-bottom.ts   # scroll behavior (see below)
  components/
    chat-widget.tsx     # the composed default (launcher + panel, header, expand/minimize)
    launcher.tsx        # floating button (chat bubble ⇄ chevron)
    parlant-logo.tsx    # the brand bird mark
    message-list.tsx    # scroll container + preface + "new messages" pill
    message-bubble.tsx  # a message: reveal text, tool widgets, reasoning disclosure, attachments
    streaming-text.tsx  # the Markdown + token-reveal engine (shared by replies & reasoning)
    composer.tsx        # textarea + inline attach + Send/Stop icons
    status-bar.tsx      # the glowing thinking line + live-streamed narration
    tool-widget.tsx     # renders tool result.metadata via a small registry
    link-tagger.tsx     # hands the utm setting to everything that renders a link
    attachment.tsx      # inbound/outbound file references
```

Hooks are headless (no styling); components are presentational and restyleable. The default
`ChatWidget` just wires them together — replace it with your own composition anytime.

## Behavior decisions (and why)

These are the non-obvious UX calls, baked in with knobs to change them.

### Streaming animation

Agent text arrives from the model in uneven bursts, so dropping raw chunks straight into the DOM
reads as choppy. `StreamingText` (in `message-bubble.tsx`) instead **buffers** the incoming text
and reveals it at a steady `streamSpeed`, wrapping each newly shown unit in the `streamStyle`
animation — decoupling display pace from network pace is what makes it feel deliberate. Already
revealed text never re-animates, and there's intentionally no blinking cursor.

Pick the feel with two props (defaults **`streamStyle="blur"`**, **`streamSpeed="normal"`**):

| `streamStyle` | Reveal |
|---|---|
| `instant` | No animation — text appears exactly as it streams (original behavior). |
| `fade` | Each word fades in. |
| `rise` | Each word fades and lifts slightly into place. |
| `blur` (default) | Each word resolves out of a soft blur (dreamier; can hurt legibility on long replies). |
| `letter` | Each character fades in (smoothest / most granular). |
| `warm` | Each word arrives dim and warms to full ink — brightness only, no motion. |

`streamSpeed` is `"slow" | "normal" | "fast"` or a raw characters-per-second number. All motion
respects `prefers-reduced-motion`. Only live agent streams animate — history and the customer's
own messages render at once.

### Markdown

Agent replies (and streamed reasoning) render as **Markdown**, and the reveal animation runs
*through* it — a bold list item streams in word-by-word with `rise`/`blur`/etc. intact. This is
a tiny zero-dependency renderer (`lib/markdown.ts` + `streaming-text.tsx`), not react-markdown:
owning the renderer is what lets us wrap each text token in an animation, which a black-box
Markdown tree can't. It closes unterminated syntax mid-stream (a half-typed `**bold` renders
bold immediately instead of flashing raw asterisks), and on completion/resume you're left with
proper formatted Markdown.

Supported: **bold**, *italic*, `inline code`, links, headings, bullet/ordered lists,
blockquotes, and fenced code blocks. Not (yet) supported: tables, task lists, images, and syntax
highlighting — deliberately dropped to stay dependency-free. Customer messages render as plain
text.

### Scrolling

`use-stick-to-bottom` keeps the view pinned to the latest message **only while you're already
at the bottom**. Scroll up to read history and auto-scroll stops; a "↓ New messages" pill
appears and snaps you back on click (or when you send your own message). Streaming uses
instant scroll (smooth-scroll janks token-by-token); discrete new messages scroll smoothly.

### Status: thinking vs. typing

The status line is derived from a **phase**, not raw status events, so it's robust to a
late/missing `ready`:

- **thinking…** while the agent is working and no reply text has appeared yet (covers the
  `processing`/tool phases). A soft band of light sweeps through the line while it shows.
  With `showReasoning` (the default), the line is the agent's actual narration ("Thinking",
  a tool's own narration, …) — a **transient** line, not a persistent chain-of-thought.
  Tool narrations and the engine's own labels reach every caller. Only the model's chain
  of thought is gated by the server's `read_event_reasoning` permission (guests and
  customers lack it under the default production policy); they see "Thinking" in its place.
- **(nothing)** while the reply streams — the growing text is the indicator.
- **cleared** the moment the reply finishes, even if the server's `ready` status is late.

### Processing & compaction

While a turn is processing, the composer isn't disabled — the send button becomes **Stop**
(Parlant supports interrupting: cancelling reprocesses the turn). The composer is only disabled
before the session is ready or while the server is unreachable (see below). Compaction (session
summarization) runs *after* the reply is delivered, so it never blocks input; the widget simply
**filters out** the compaction summary message (`metadata.__parlant__.source === "compacter"`)
so it doesn't render as a bubble.

### Server reachability

`use-server-health` polls `GET /healthz` (every 20s while healthy, every 5s while down). When
the server is unreachable the widget shows a **"Can't reach the server — reconnecting…"** banner
and disables the composer; both clear automatically when it recovers. That banner is the single
source of truth for connectivity — the transient error line is suppressed while offline and clears
itself once the stream delivers an event again, so you never get a stale "Something went wrong"
stuck on screen. Independently, the event stream reconnects from its last offset on any drop.
Failed reconnects back off from 1 second, doubling up to 30 seconds, and a failed session start
is retried the same way. Customer-authenticated requests refresh the token and retry once on a
401. If a persisted guest credential expires, the widget clears that guest's stale sessions,
starts a new anonymous conversation, and tells the visitor what happened instead of retrying
forever. A saved conversation the server refuses (403) — for example one started by a guest
identity this browser no longer holds — is restarted the same way, as is one saved without a
guest token.

## Rich content

A tool can return a **typed payload the widget renders as custom UI** — an order card, a chart,
a map — instead of (or alongside) plain text. It's three moving parts: your **tool** emits a
widget, you **register a renderer** for its type, and the widget **joins it to the turn** and
renders it. The bundled sample agent (`sample-agent/agent.py` + `example/src/widgets.tsx`) is a
complete working example of exactly this.

### 1. Emit a widget from your tool (server, Python)

A `ToolResult` has two channels: **`data`** is what the *model* sees (so the agent can talk
about it), and **`metadata`** is **frontend-only** — never sent to the model. Put your widget in
`metadata` under a `widget` key:

```python
@p.tool
async def get_order_status(context: p.ToolContext, order_id: str) -> p.ToolResult:
    order = await my_store.fetch_order(order_id)
    return p.ToolResult(
        # Model-visible — lets the agent reference the order in its reply:
        data=f"Order {order_id}: {order.status}, arriving {order.eta}.",
        # Frontend-only — the widget the UI will render:
        metadata={
            "widget": {
                "type": "order_card",       # ← matches a registered renderer
                "orderId": order_id,
                "status": order.status,
                "eta": order.eta,
            }
        },
    )
```

> Keep `metadata` small — it's serialized into the tool result (which has a size cap). Send a
> **URL/reference**, never bytes (don't inline an image as a `data:` URL, for example).

### 2. Register a renderer for that `type` (frontend)

`registerWidget(type, renderer)` maps a widget `type` to a React component. The renderer
receives the widget object — its `type` plus whatever props your tool sent (and, for widgets that
answer in the chat, a context; see [Interactive renderers](#interactive-renderers)):

```tsx
import { registerWidget } from "@/parlant-chat";

registerWidget("order_card", ({ orderId, status, eta }) => (
  <div className="rounded-xl border bg-muted/30 p-3 text-sm">
    <div className="font-medium">Order {orderId}</div>
    <div className="text-muted-foreground">{status} · arriving {eta}</div>
  </div>
));
```

Registration must run **once, before the widget mounts**. Put all your renderers in one file and
import it for its side effect at your app entry:

```tsx
// widgets.tsx — your renderers
registerWidget("order_card", …);
registerWidget("weather_card", …);

// app entry (e.g. main.tsx), above where <ChatWidget/> renders
import "./widgets";
```

### 3. How it reaches the screen

You don't wire anything else — the widget already does the plumbing:

1. Your tool's result rides the turn's **`tool` event** on the same SSE feed the widget consumes.
2. `use-parlant-stream.ts` pulls `tool_calls[i].result.metadata` off that event into a
   `ToolWidget`, tagged with the turn's **`trace_id`**.
3. `message-list.tsx` joins it to the agent reply of that same turn (matching `trace_id`) and
   renders it **above the reply** via `ToolWidgetView` (`tool-widget.tsx`), which looks your
   `type` up in the registry.
4. An **unregistered `type` falls back to a JSON preview**, so nothing breaks while you build —
   you'll see the raw payload until you register a renderer for it.

### Metadata conventions

`ToolWidgetView` understands three keys in `result.metadata`:

| Key | Shape | Renders as |
|---|---|---|
| `widget` | `{ type, ...props }` | one widget |
| `widgets` | `[{ type, ...props }, …]` | several widgets, in order |
| `attachments` | `[AttachmentRef, …]` | file references, via the built-in `Attachment` component |

### Built-in: `choices`

One widget type is built in: **`choices`**, a "pick one" question. The options render as a row
of buttons above the agent's reply, and clicking one sends it as the visitor's own chat message
— through the same path as the composer, so it shows in the transcript and the agent reads it
like any other message. No renderer to register:

```python
return p.ToolResult(
    data="Offered Tue, Nov 3 and Wed, Nov 4.",
    metadata={
        "widget": {
            "type": "choices",
            "id": "meeting-dates",               # optional, stable per kind of question
            "prompt": "Which day suits you?",    # optional, the button group's accessible name
            "options": [                         # required, 1–8 options
                {"id": "2026-11-03", "label": "Tue, Nov 3"},
                {"id": "2026-11-04", "label": "Wed, Nov 4", "reply": "Wednesday, Nov 4 works"},
            ],
        }
    },
)
```

| Field | Required | Meaning |
|---|---|---|
| `options` | yes | 1–8 options. Each has a unique `id` and a `label` (the button text). |
| `options[].reply` | no | The message sent when the option is clicked. Defaults to `label`. |
| `prompt` | no | Names the button group for screen readers. Not shown, since the agent's reply asks the question. |
| `id` | no | A stable name for the kind of question (e.g. `meeting-dates`), exposed as `data-choices-id`. |

Clicking sends `reply ?? label` once. After that:

- the chosen option shows as selected (`aria-pressed`) and the whole group is disabled;
- the group is also disabled as soon as any later customer message exists, typed or clicked, so
  stale options can't be clicked;
- in a resumed conversation, past options render disabled, and the chosen one is marked when its
  `reply ?? label` matches the next customer message.

A multi-step pick is one `choices` widget per step: a tool returns the dates, and once the
visitor picks one, the next tool call returns the time ranges on that date.

Metadata that doesn't fit (no options, more than 8, an option without an `id` or `label`, or
duplicate ids) renders nothing for that widget rather than breaking the turn. To render choices
your own way, register a `choices` renderer; it replaces the built-in one.

### Interactive renderers

A renderer gets a third argument, a context for widgets that act in the conversation:

| Field | Meaning |
|---|---|
| `sendMessage(text)` | Sends `text` as the visitor's message, exactly as if they had typed it. |
| `isLatest` | `true` until a customer message comes after this widget's turn. Use it to disable stale controls. |
| `nextCustomerMessage` | The text of that later customer message, if any (e.g. to mark an earlier answer). |
| `appendUtm(href)` | `href` with the widget's `utm` parameters added, under the same rules as the agent's links. Unchanged when no `utm` is set. |

```tsx
registerWidget("rating", ({ question }, _widget, { sendMessage, isLatest }) => (
  <div role="group" aria-label={question}>
    {[1, 2, 3, 4, 5].map((n) => (
      <button key={n} type="button" disabled={!isLatest} onClick={() => sendMessage(`${n} stars`)}>
        {n}★
      </button>
    ))}
  </div>
));
```

Renderers that take only `(props, widget)` keep working unchanged.

## Link attribution (UTM)

Pass `utm` to tag every link the agent writes, so your analytics can tell which visits came
from the chat. Give the keys without the `utm_` prefix:

```tsx
<ChatWidget
  serverUrl="…" agentId="…"
  utm={{ source: "site-chat", medium: "chat-widget", campaign: "spring" }}
  utmLinks="external"
/>
```

- **Which links.** Every `http:`/`https:` link in an agent reply, whether written as Markdown
  (`[pricing](https://…)`) or as a bare URL. `mailto:`, `tel:` and relative links are left
  alone, and so are attachments (often signed storage URLs that an extra parameter would break)
  and the "Powered by Parlant" link, which carries Parlant's own attribution.
- **`utmLinks`.** `"all"` (default) tags every such link. `"external"` leaves links to the
  embedding page's own host unchanged. Analytics tools such as Google Analytics restart a
  visitor's attribution when they follow a UTM-tagged link within the same site, so prefer
  `"external"` when the agent mostly links to your own pages. Hosts are compared exactly:
  `www.example.com` and `docs.example.com` are different hosts.
- **Existing parameters win.** A `utm_*` key the link already has is kept; only the missing
  keys are added. The rest of the link's query and its `#fragment` are kept as written.
- **Keys.** `source`, `medium`, `campaign`, `term`, `content` and `id`. Empty values are skipped.
- Only the rendered link changes. The message text, as stored in Parlant, stays as written.

Custom widgets can tag their own links the same way with `context.appendUtm(href)` (see
[Interactive renderers](#interactive-renderers)). The pure `appendUtm(href, utm, { links })`
function is exported too, for links you render outside the widget.

## Reacting to session labels

A session gets labels when a policy or observation declared with `labels=[…]` matches, or when
your code calls `session.labels.add(…)`. Pass `onLabel` to act on them in the page, e.g. to
record a conversion when the agent labels a session `lead`:

```tsx
<ChatWidget
  serverUrl="…" agentId="…"
  onLabel={(label, { sessionId, labels }) => {
    if (label === "lead") recordLead(sessionId);
  }}
/>
```

- It is called **once for each label a turn adds**, with the session id and the session's full
  current labels.
- A reopened conversation's existing labels are not reported, and neither are removals.
- The widget reads the session's labels after each turn ends, so the call comes then, not while
  the reply is still being written. Labels changed outside a turn (e.g. from your backend) are
  reported after the next turn.
- An exception thrown by your callback is logged and ignored; the conversation carries on.
- Act only on the labels you expect. A session can carry internal labels too, and the callback
  sees all of them.
- The widget sets no cookies. If you store something (a cookie, an analytics event), do it in
  your callback, under your site's consent rules.

## Attachments

The composer's paperclip calls an integrator-supplied `onUpload` (you upload to your own storage
and return an `AttachmentRef`), then posts the file on the message. Parlant stores no bytes — you
own the storage. The attach button only appears when `onUpload` is set; the widget shows the
pending/uploading state while your promise resolves. Restrict the picker with `acceptFileTypes`
(e.g. `"image/*"`).

```tsx
<ChatWidget
  serverUrl="…" agentId="…"
  onUpload={async (file) => ({ url: await uploadToMyBucket(file), media_type: file.type })}
  acceptFileTypes="image/*"
/>
```

A tool can then read that attachment (its URL lives on the customer's message) and act on it —
which is exactly what the sample agent's `describe_attached_image` does before returning a
`picture_description` widget.

## Customizing

- **Theme** — pass `theme="…"` for a built-in palette: `monochrome` (default; light/dark
  follows the OS), `parlant` (brand green), `ocean` (dark), `amethyst` (dark), `sand` (light).
  Each is a set of shadcn-style CSS variables on `.parlant-widget` — `--background`,
  `--foreground`, `--primary` / `--primary-foreground`, `--muted` / `--muted-foreground`,
  `--border`, `--ring`, `--destructive`. Reskin by redefining them (no Tailwind needed), or add
  your own theme by defining `.parlant-theme-<name>` in `styles.css` and passing that name.
  Every element also carries a `parlant-*` class hook (`parlant-widget`, `parlant-panel`,
  `parlant-msg`, `parlant-composer`, …) so you can restyle any part directly.
- **Transport** — everything network lives in `parlant-chat/lib/transport.ts`; swap it (e.g.
  for the official `parlant-client`) without touching the UI.
- **Tool widgets** — register renderers by `type` (see above); `choices` is built in and can be
  overridden the same way.
- **Composition** — ignore `chat-widget.tsx` and build your own surface from the hooks and
  primitives.

## `ChatWidget` props

| Prop | Type | Default | Purpose |
|---|---|---|---|
| `serverUrl` | `string` | — (required) | Parlant server base URL. |
| `agentId` | `string` | — (required) | The agent to converse with. |
| `mode` | `"floating" \| "inline" \| "input" \| "bar"` | `"input"` | Launcher bubble, embedded in your layout, an input box docked in the corner, or a message bar centered at the bottom of the screen. `input` and `bar` are the same surface in two positions: both start collapsed and grow into the full panel on the first message. |
| `auth` | `{ type: "guest" } \| { type: "customer"; identityKey: string; getToken: () => Promise<string> }` | `{ type: "guest" }` | Select guest auth or supply a stable customer persistence key and customer-token callback. |
| `onUpload` | `(file: File) => Promise<AttachmentRef>` | — | Enables the attach button (BYO storage). |
| `showAttachments` | `boolean` | `true` | Show the attach button (only matters together with `onUpload`). |
| `acceptFileTypes` | `string` | — | Restrict the attach picker (HTML input `accept`, e.g. `"image/*"`). |
| `showReasoning` | `boolean` | `true` | Show the agent's narration as the thinking line, streamed live (Markdown, animated). Callers the server doesn't let read reasoning see a generic "Thinking…". |
| `reasoningMode` | `"transient" \| "collapsible"` | `"transient"` | Once the reply arrives: reasoning disappears, or folds into a "Thought for Xs" toggle. |
| `errorMessage` | `string` | `"The agent ran into an error…"` | Inline notice shown at the end of the transcript when a turn fails (session `error` status). |
| `title` | `string` | agent's name | Header title; defaults to the agent name from `GET /agents/{id}`. |
| `preface` | `string` | — | Muted intro line shown above the conversation at session start. |
| `disclaimer` | `ReactNode` | — | Small print under the input (accepts JSX, e.g. a privacy-policy link). A "Powered by Parlant" line always shows beneath it. |
| `launcherHint` | `string \| null` | `"Talk to us"` | Jumping nudge (plain text + arrow) above the launcher before its first open; `""`/`null` hides it. |
| `launcherHintColor` | `string` | `"#000000"` | Color of the launcher hint text + arrow (any CSS color). |
| `showNewChat` | `boolean` | `false` | Show a "New chat" header button (with confirm) that starts a fresh session. |
| `streamStyle` | `StreamStyle` | `"blur"` | How streamed agent text reveals — see [Streaming animation](#streaming-animation). |
| `streamSpeed` | `StreamSpeed` | `"normal"` | Reveal pace: `"slow" \| "normal" \| "fast"`, or characters/second. |
| `placeholder` | `string` | `"Type a message…"` | Text shown in the empty message input, in every mode. |
| `openMode` | `"expanded" \| "sidebar"` | `"expanded"` | How the chat opens: a regular panel or a full-height sidebar. Mobile sidebar fills the screen. |
| `persistentSuggestions` | `string[]` | — | Chips retained until selected once per session; return for a new chat. |
| `maxVisibleSuggestions` | `number` | `3` | Maximum visible chips, taking opening suggestions before unchosen persistent ones. |
| `openingSuggestions` | `string[]` | — | Conversation starters shown as chips; clicking one sends it. They appear only before the conversation begins: while the card is collapsed in `input`/`bar`, and while the open transcript is empty in `floating`/`inline`. |
| `initialMessage` | `string` | — | Sent as the customer's first message, in a new conversation started for it. See [Opening with a message](#opening-with-a-message). |
| `customerSound` | `boolean` | `true` | Play a short blip when the customer's message is sent. |
| `agentSound` | `boolean` | `true` | Play a short blip when an agent reply arrives. |
| `soundTheme` | `SoundTheme` | `"velvet"` | Which pair of sounds to play — 13 built in, see [Sound themes](#sound-themes). |
| `utm` | `UtmParams` | — | UTM parameters added to the links the agent writes. See [Link attribution (UTM)](#link-attribution-utm). |
| `utmLinks` | `"all" \| "external"` | `"all"` | Which links `utm` applies to: all, or only links to a host other than the page's own. |
| `onLabel` | `(label, { sessionId, labels }) => void` | — | Called once for each label a turn adds to the session. See [Reacting to session labels](#reacting-to-session-labels). |

## Sound themes

The widget plays a short blip when a message is sent and another when a reply arrives. Both
are synthesized with the Web Audio API — no audio files, nothing to host — and `soundTheme`
picks which pair. Each theme sounds different in each direction, so the two are
distinguishable without looking:

| Theme | Sounds like |
|---|---|
| `velvet` (default) | A soft chime over a deep, gently swelling thud and fading echoes. Customer messages use the sent cue; every agent message uses the fuller first-incoming cue. |
| `glass` | High, thin sines with a faint overtone — a struck wine glass. |
| `drop` | One round low sine. Unhurried and easy to ignore. |
| `marimba` | Wooden: a note with its octave stacked on top and a fast decay. |
| `thock` | Deep and soft, like a keyboard switch. Closer to felt than heard. |
| `tabla` | A drum-like pitch bend — down as the message leaves, up as one arrives. |
| `snap` | Fingers snapping: a noise tick, with a pitched ping inside the reply. |
| `chip` | Bright, plastic triangle pairs. Unmistakably digital. |
| `bloop` | A fat sine glide, falling out and rising in. |
| `anvil` | Struck metal: a short square with a ringing tail above it. |
| `chalice` | A slow bloom of notes; the reply adds a third. The most ceremonial. |
| `shard` | Electric: a high-passed noise tick with a glassy ping riding on it. |
| `halo` | Stacked high sines arriving in sequence — a shimmer rather than an event. |

```tsx
<ChatWidget serverUrl="…" agentId="…" soundTheme="marimba" />
```

Turn either direction off with `customerSound={false}` / `agentSound={false}`. To add your own,
extend `SOUND_THEMES` in `lib/sounds.ts` — a theme is two short lists of notes (frequency,
duration, gain, and optionally a start offset, a glide target, and a source). The source is any
oscillator shape, or `"noise"` for a burst of white noise high-passed at `freq`, which is what
gives a tick an electric, unpitched character instead of sounding like a chirp.

## Session persistence

By default the widget **remembers the conversation across reloads in the same browser**. On
session bootstrap it stores the session id — and, for guests, the guest token — in the browser's
`localStorage`. Versioned keys are scoped by server, agent, and auth identity:

```
parlant:v2:guest:<encodedServerUrl>      → {"token": "…", "sessions": {"<agentId>": "<sessionId>"}}
parlant:v1:session:<encodedServerUrl>:<encodedAgentId>:customer:<encodedIdentityKey>
```

A visitor has **one guest identity per server**, shared by every agent and every tab of your
site, and each agent keeps its own conversation with it. The conversations live inside the
identity's record, so when the identity is replaced (its token expired) they are all dropped
together: no widget is left resuming a conversation its token can't read.

Several widgets and tabs work together:

- A new visitor who opens several agents or tabs at once gets one guest identity, and one
  conversation per agent. Creating them is serialized across tabs with the Web Locks API
  (`navigator.locks`), or across the widgets of one page in browsers without it.
- Every request reads the current token, so a renewal or a new identity from one tab is used by
  all of them.
- **New chat** in one tab moves other open tabs of the same agent to the new conversation. When
  the identity is replaced, every open tab starts over and says so.

Customer sessions are additionally scoped by `identityKey`, so logging out or switching accounts
cannot resume another identity's locally cached session. Customer bearer tokens remain
memory-only. The **New chat** button (`showNewChat`) clears only the active session id; it keeps
the guest identity. In a non-browser environment (SSR) where `localStorage` is undefined, the
transport falls back to no persistence.

Version 1 keys (`parlant:v1:guest:*` and `parlant:v1:session:*:guest`) are migrated on first
read; a v1 guest conversation saved without a token is dropped, since nothing can resume it.

**This is not a `ChatWidget` prop.** The store is set one level down, on the transport, via
`TransportOptions.storage` — anything matching `Pick<Storage, "getItem" | "setItem" | "removeItem">`.
Because this is copy-paste code, you configure it by passing `storage` where `chat-widget.tsx`
calls `useParlantSession({ serverUrl, agentId, auth })` (or by using the hook directly in
your own surface):

```ts
// Per-tab instead of per-browser: forgotten when the tab closes.
useParlantSession({ serverUrl, agentId, auth, storage: sessionStorage });
```

```ts
// Disable persistence entirely — a brand-new session on every load (in-memory shim).
const memory = new Map<string, string>();
useParlantSession({
  serverUrl,
  agentId,
  auth,
  storage: {
    getItem: (k) => memory.get(k) ?? null,
    setItem: (k, v) => void memory.set(k, v),
    removeItem: (k) => void memory.delete(k),
  },
});
```

The key construction lives in `ParlantTransport` and `lib/guest-identity.ts`. `storage` swaps the backing store without
weakening the server/agent/identity isolation built into those keys. Older unversioned
`parlant:session:*` and `parlant:guest:*` entries are intentionally ignored.

### Opening with a message

When your page has already asked the visitor something, for example a form that opens the
chat, pass it as `initialMessage`. The widget starts a **new** conversation instead of resuming
the stored one (the guest identity is kept, as with **New chat**), and sends the text as the
customer's first message as soon as the session is live. In `input` and `bar` modes it also
opens the panel.

```tsx
<ChatWidget
  serverUrl="https://your-server"
  agentId="sales"
  mode="inline"
  initialMessage={`My email is ${email} and I'd like to learn about Parlant for our company.`}
/>
```

It's sent once per mount, so pass it only on the visit it's for. On a later visit, render
without it and the conversation resumes as usual.

## Not included (by design)

- **File storage** — you provide `onUpload`; Parlant carries references, not bytes.
- **Auth key handling** — the widget only holds a guest or per-customer token; you mint
  customer tokens on your backend (see [Authentication](#authentication)).
- **A build/publish step** — this is source to copy, not a package to install.

## Development

The dev harness (Vite; the demo page itself uses Tailwind, but the widget does not) lives at
the `widget/` root; the demo pages live in
`example/src/` — **`guest.tsx`** and **`authenticated.tsx`**. The playground lets you set the
**server URL, agent id, customer identity key, and customer token live in the UI** (the `.env`
values below are just the initial defaults), plus toggles for display mode and stream style/speed.
Its signed-in mode is a manual pasted-token harness, not an example token-minting endpoint.

```bash
cd widget
npm install
cp .env.example .env    # optional: prefill VITE_PARLANT_URL and VITE_AGENT_ID
npm run dev             # opens the playground
npm run typecheck       # tsc over parlant-chat/ and the demo
npm test                # transport and React session/auth tests
```

The copy-paste target is the single `parlant-chat/` folder. Everything else (`index.html`,
`example/`, the config files) is just the harness for developing and demoing it.

### Sample agent

`sample-agent/agent.py` is a runnable Parlant agent — **"Agent Tom"** (id `tom`) on port
**8810** — that chats normally and **describes any image you attach**, using an OpenAI vision
model. It's a dev-mode server (no auth), so the widget's guest flow works against it directly.

```bash
# from the repo root
OPENAI_API_KEY=sk-... uv run widget/sample-agent/agent.py
```

The playground already defaults to `http://localhost:8810` / agent `tom`, so just click
**Connect**, attach a photo, and ask what's in it. How it works: Parlant only shows the engine a
text note for an attachment, so the agent's tool pulls the attachment's URL off the customer
message and calls the vision model itself
(`p.Session.current.interaction.last_customer_message.attachments`). The demo
uploader returns a `data:` URL, so this works locally with no external storage.
