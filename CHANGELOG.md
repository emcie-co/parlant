# Changelog

All notable changes to Parlant will be documented here.

This file starts over at Parlant 4. Parlant 4 reshapes enough of the public
vocabulary that the older entries describe an API you can no longer call, so
keeping them here would mislead more than it informs. Everything up to 3.3.2 is
in this repository's history — read it at the `v3.3.2` tag, or with
`git show v3.3.2:CHANGELOG.md`.

Each entry says what changed and what it means for you, under `Added`,
`Changed`, `Deprecated`, `Removed`, `Fixed` or `Security`.

## [Unreleased]

### Added

- The chat widget supports `openMode="sidebar"`: a full-height column beside
  the page that widens on expand and fills the screen on mobile. Minimizing
  restores page space without conflicting inner-panel width animations.
- `persistentSuggestions` keeps prompts available until selected once per
  session. `maxVisibleSuggestions` defaults to 3 and prioritizes opening prompts.

- Tools can report incurred USD costs by category with `ToolResult.costs`.
  `Session.emit_tool_result()` records completed background work before a
  separate `Agent.utter()` follow-up. Tool costs appear in session usage and
  count toward session budgets, including when no follow-up is delivered.

- `parlant init --scaffold lead-gen` generates a website lead-generation agent:
  it answers visitors' questions, qualifies them in a short procedure, scores
  them in code, and books meetings with a rep in the chat. It runs on
  in-memory fakes for the CRM, calendar, mailer and sales inbox until you
  connect your own.

### Changed

- The chat widget now defaults to the Focus pull (`blur`) text reveal.

- Rename the widget's `suggestions` prop to `openingSuggestions`; update embeds
  that pass conversation starters. Agent replies now use the full chat canvas,
  with more paragraph spacing and improved wrapping. Customer bubbles and the
  input have more padding; avatars, customer bubbles and send controls use 10px
  corners. Collapsed input controls have less space below them, and the header
  controls row no longer has a divider.

- The chat widget now defaults to the new `velvet` sound theme: a soft chime
  layered over a deeper thud with a gentle attack and fading echoes. Customer
  messages use a short sent cue; every agent reply uses the fuller incoming cue.
  Existing themes remain available through `soundTheme`.

- Tool narrations and Parlant's own status labels ("Running tools",
  "Reviewing message", ...) reach every caller, including guests and customers
  under the default production policy. Only the model's own reasoning is gated
  by `read_event_reasoning`; without it, that status event arrives with
  `message` set to `"Thinking"` and no `chunks`. The event's metadata flag is
  now `__parlant__.internal_reasoning` (was `reasoning`). A client that
  special-cased the old `"[redacted]"` stub can drop that code.
- In the chat widget's `reasoningMode="collapsible"`, "Thought for Xs" opens
  to every status the agent showed during the turn, in order, rather than the
  last one alone. The model's reasoning appears among them when the caller
  may read it.

- The `retail` scaffold is now `retail-store`. Run
  `parlant init --scaffold retail-store`; `--scaffold retail` is no longer
  accepted. Projects already generated are unaffected.

- The chat widget shows the agent's narration while it thinks, by default. The
  line under the messages now reads "Thinking", or a tool's own narration, with
  a soft band of light sweeping through it, in place of the bouncing dots.
  `showReasoning` defaults to `true`; pass `false` to keep narration off.

- The chat widget's `suggestions` now show in `floating` and `inline` modes too,
  as starter chips above the input, for as long as the open conversation has no
  messages. They used to appear only in `input`/`bar` mode, where they are still
  tied to the collapsed card. Nothing changes for a widget that passes no
  `suggestions`; one that passes them and runs in `floating` or `inline` starts
  showing them.
- Gemini's default embedding model is now `gemini-embedding-2`, which takes
  8192 input tokens instead of `gemini-embedding-001`'s 2048, at the same 3072
  dimensions. Stored embeddings are kept per embedder, so an existing Gemini
  project re-embeds its documents once on its next start. To stay on the old
  model, route `embedding.*` to `gemini:gemini-embedding-001`.
- On Gemini, agents at low or medium effort now generate responses with
  `gemini-3.5-flash-lite` at medium thinking instead of low. At low thinking
  the model often wrote its deliberation into the reply ("The user is asking
  about…"). Responses cost about 50% more and start about a second later. To
  keep the old setting, route `react[effort>=low,effort<high]` to
  `ModelConfig("gemini:gemini-3.5-flash-lite", reasoning="low")`.
- The `AGENTS.md` that `parlant init` writes now asks the coding agent to have
  the user sign in with `parlant dashboard auth login` and then check the
  project itself; maps each kind of change to its construct (tone of voice and
  wording choices belong to canned responses, for example); explains hosting
  several agents on one server; and says that an edit which fails at startup
  stops the server under hot reload.
- That `AGENTS.md` now opens with a rule: run `parlant docs query` before
  writing or changing any Parlant SDK call or server configuration, however
  small, and even when copying from another project, because a coding agent's
  training data describes earlier Parlant versions. It also tells the coding
  agent to recommend that the user let it save that rule as a memory, so it
  carries into later sessions and other Parlant projects. Projects scaffolded
  earlier keep their `AGENTS.md`; copy the section over by hand.

### Fixed

- Background `Agent.utter()` follow-ups now record their own model usage and
  costs instead of reusing an already finalized trace. Completed background
  tool results and follow-ups retain their originating turn reference, even
  after a newer customer message arrives, so Dashboard integrations can group
  them without counting a charge twice. Repeated follow-ups also preserve the
  caller's current session and agent context.

- A caller without `read_event_reasoning` could still read the model's
  reasoning: only the status event's `message` was stubbed, while `chunks`
  carried the full text. Both are now withheld.

- An agent's own model routes keep their `reasoning`. Routes passed to
  `create_agent(models=...)` or `agent.update(models=...)` are stored, and the
  store dropped `reasoning`, so those routes fell back to the engine's
  effort-based default. Server routes were unaffected.

- Background tool costs now use the current session store, including after
  store configuration changes. Completed paid work no longer fails with
  `ItemNotFoundError` before its follow-up can be sent. On managed hosting,
  accounting follows the same persisted runtime route as the emitted result,
  rather than the separate SDK declaration store. Session labels also use this
  route, so a background delivery failure can label the hosted session.

- Canned responses are translated into the conversation's language. An
  approved response written in English used to go out word for word to
  visitors writing in German or French; the responder now translates it
  faithfully, keeping its numbers and links, and the reviewer accepts the
  translation.
- The first Gemini Interactions request initializes Google's client off the
  event loop, so its cold imports no longer hold up other sessions or health
  checks. Concurrent requests share initialization; cancelling one does not
  interrupt another.
- The `basic` scaffold's agent instructions no longer run two words together
  ("meantto").
- The docstrings for a tool's `consequential` flag said it was unused. A
  consequential tool is flagged to the model as such, and a proposed call to
  it triggers tool review.
- `parlant dashboard auth login` now prints the complete approval link, including
  the request code. Opening the printed link works when the browser cannot open
  automatically or when using `--no-browser`.
- Gemini turns that recall tools no longer fail with `404 NOT_FOUND` for new
  Google users. Token counting for Gemini embedders used `gemini-2.5-flash`,
  which Google no longer serves to new users; each embedding model now counts
  its own tokens.
- When indexing fails, the error now names the step and the provider's own
  error, such as `RuleRelativityProposer failed after 3 attempts: ClientError:
  429 RESOURCE_EXHAUSTED…`, and says to check the provider's key and quota.
  Previously a server stopped at startup with only `Indexing failed: Indexing
  failed`.
- A hedged ReAct step now completes as soon as the faster stream finishes,
  instead of waiting for the slower one. A slow first response no longer holds
  the agent's step for as long as it takes to time out. The slower stream is
  never cancelled: its tokens are still reported against the same turn once it
  finishes, and the step counts both requests. Custom `ReactGenerator`
  subclasses get this by accepting `usage_reporter` and passing it to
  `super().__init__()`; without one, a step still waits for the slower stream.
