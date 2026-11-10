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

### Fixed

- MCP tools that return structured results, such as a pydantic model, now give
  the agent the result as plain JSON. Before, the call ran but the agent was
  told "Tool call error", so it could retry a call that had already taken
  effect or tell the customer it failed.
- Gemini agents under concurrent load no longer queue behind each other while
  a prompt cache is being created. Requests for other caches, and requests whose
  cache already exists, go ahead at once; requests for the same cache still
  share one creation, so it is created and its storage charged once.
- Servers with MongoDB stores start faster as session and event history grows.
  When a store's recorded version already matches the running one, startup no
  longer reads every document back through its loader. When a migration is
  needed, documents are loaded and migrated as before.

## [4.0.4] - 2026-11-08

### Added

- `Agent.queue_response()` combines background follow-ups with other ready
  results or a customer turn. Pass `after=coroutine` to register work before it
  starts; only the deferred reply waits for the registered batch. Coordination
  lives in one process and requires session routing affinity; pending work is
  not recovered after a restart. Queued replies tolerate history edits and
  terminate interrupted streams without appending stale text.

### Changed

- When tool preambles are encouraged, the agent writes them, and the progress
  updates that follow, in the customer's language.
- Agents acknowledge recurring information gaps when different questions cannot
  be answered, without treating a new question as one already answered.
- Agents ask which record or request the customer means when it's ambiguous,
  and which of several unfinished requests to handle first, before collecting
  task details.

### Fixed

- Gemini conversations can resume from background results after history is
  cleared, without an invalid tool-response ordering error.
- The Parlant dashboard now shows your agent's reasoning under Agent activity
  when the agent streams its messages, the default for `create_agent`. Each
  step's reasoning is now recorded as a `loop.reasoning` trace event. Before,
  it reached the dashboard only as a log line, which the dashboard read only
  from agents in block output mode.
- `parlant[anthropic]` works with the current Anthropic SDK. A fresh install
  picked anthropic 1.0 or later, whose client refused Parlant's request timeout,
  so no Claude model could start. The extra now requires `anthropic>=1.0.0`;
  if your project capped anthropic below 1.0 to work around this, drop the cap.
- The chat widget lays out a right-to-left reply, such as Hebrew, right to left
  even when it opens with an English word, a URL or code. It used to take the
  direction from the first letter, so such a reply read left to right with its
  punctuation on the wrong side. Each paragraph, the message box and the
  suggestion buttons now follow the script most of their words are in.

## [4.0.3] - 2026-11-05

### Changed

- `parlant dashboard` commands now tell the Parlant dashboard which command
  sent each request. Every request carries `User-Agent: parlant-cli/<version>`,
  the command's name in `X-Parlant-CLI-Command` (for example
  `parlant dashboard session list`, never its arguments or option values),
  and an `X-Parlant-CLI-Invocation` id shared by every request from one run.
  The dashboard uses them to show your CLI activity as the commands you ran.

## [4.0.2] - 2026-11-05

### Added

- The chat widget lets the AI agent in a visitor's browser talk to your agent
  over WebMCP (`document.modelContext`; in Chrome behind a flag today). On
  every page it's on, the widget registers an `ask_agent` tool described with
  the agent's name. A call starts the widget's conversation if there is none
  yet, as one an agent holds for the customer (`customer_represented_by_an_agent`),
  sends the question through the chat, so it shows there with the reply, opens
  the chat, and answers the browser agent once the turn has ended. Follow-up
  calls continue the same conversation. The message carries
  `metadata.parlant_widget.asked_by = "customer_agent"`, so the chat and your
  tools can tell the visitor's agent from the visitor. Nothing happens in a
  browser without WebMCP; `webmcp={false}` turns it off. `ParlantTransport`'s
  `sendMessage` takes an optional metadata argument and `ensureSession` an
  option for the flag.
- `Server(session_store=..., customer_store=...)` accepts an `s3://bucket`
  URL to keep sessions and customers in Amazon S3. Give Parlant a bucket of
  its own. Credentials come from the AWS credential chain; add `?region=` or,
  for S3-compatible storage, `?endpoint=`. The storage must support
  conditional writes, which keep concurrent servers from overwriting each
  other's updates. Empty collections survive restarts, and a failed migration
  write leaves the original data in place. Any query other than a lookup by ID,
  such as listing sessions, reads every object in the collection, so use
  MongoDB if you need indexed queries at scale.
- The chat widget (`parlant widget init`) takes the agent's picture in one
  `avatar` prop: `{ type: "logo", src }`, `{ type: "photo", src, live? }` or
  `{ type: "video", src?, poster?, live?, launcher? }`. A video avatar plays a
  short muted loop under the header, scrolls away with the conversation once
  it starts, and keeps its opened size when the chat is expanded. On the closed
  launcher, `launcher` crops it onto the face. Leave out `src` to use the
  widget's own video of an agent. `live` adds the green dot, on the video's
  corner too, red while the server can't be reached. Use an H.264 MP4 saved
  with `-movflags +faststart`; iPhone Safari stalls on WebM.

### Changed

- The chat widget's persistent suggestions stand out from its opening ones:
  outlined and lettered in the theme's primary color on the chat's background,
  instead of the grey fill.
- On phones and tablets, the chat widget closes the on-screen keyboard once a
  message is sent, whether typed or picked from the suggestions, so the reply
  isn't hidden behind it.
- With the LiteLLM provider, review routes now default to
  `gemini/gemini-3.8-flash` instead of `gemini/gemini-3.6-flash`, the same
  model the Gemini provider already uses for them.
- The `lead-gen` scaffold's agent no longer guesses when a visitor asks about
  the business, its products or their project and its product knowledge has no
  answer. It says it's a great question but hard to answer properly in the
  chat, offers to connect them with the team, asks for an email or phone
  number, and passes the question on with `request_team_follow_up`. Unrelated
  questions, existing customers' order questions and prices are handled as
  before. To make that possible, `record_lead` now accepts a lead with just a
  goal and a contact: `buying_for` and `role` are optional, and an unknown one
  scores nothing.
- `parlant dashboard project confirm` finishes a local or self-hosted project
  even when its server already holds sessions, such as from trying the agent.
  It imports them into the dashboard first, as the dashboard's Sync does, and
  then completes the setup, so the project no longer stays at "Finish setup".
  An import that takes longer than about 90 seconds stops with
  `SYNC_IN_PROGRESS`; run `confirm` again to keep waiting for the same import.
  A failed import stops with `SYNC_FAILED` and its reason.
- `parlant init --dashboard --json` and `parlant dashboard project create --json`
  for a local or self-hosted project now list `nextSteps`, ending with
  `project confirm`. The scaffolded `AGENTS.md` tells coding agents to confirm
  the project once its server is connected.
- After creating a managed project whose setup check found how to run it,
  `parlant dashboard project create` now points to
  `parlant dashboard --project ID deploy`: the first live deploy completes the
  setup. The dashboard's setup page is optional for reviewing the settings.

### Deprecated

- The chat widget's `avatarUrl`, `avatarStyle` and `liveIndicator` props, in
  favor of `avatar`. They still work when `avatar` isn't set.

### Fixed

- Gemini Argon can replay saved tool calls with their original signed thought
  blocks instead of unnecessarily falling back to result-only history.
- Filtering sessions by labels now returns complete pages and working
  continuation cursors, with consistent total counts in either sort order.
- MongoDB cursor pagination now preserves `$or` filters on later pages, so
  filtered listings no longer include records outside the requested set.
- Canned responses now retain separate search vectors for their text and
  every signal, so signals no longer collide in the relevance index.
  Applies when responses are created or updated; existing indexes are not
  rebuilt automatically.
- On iPhone, focusing the chat widget's message box no longer zooms the page
  in, which pushed the open chat off the right edge of the screen. The box
  uses a 16px font on touch screens, the size below which iOS zooms.
- Local and MCP tools now accept comma-separated list arguments without
  brackets while preserving quoted commas and empty lists.
- The chat widget's latest message no longer slips under the suggestions and
  message box when something takes the transcript's space, such as the
  "Connection interrupted" banner appearing just after a send. The transcript
  stayed pinned to the bottom only as messages grew, not as its view shrank.
- Invalid context-variable freshness schedules are rejected before saving,
  so a failed schedule update leaves the variable's existing data intact.
- A customer message that arrives while the agent is still working on the
  previous one now replaces that turn, as intended, even when the engine was in
  the middle of rejecting a step: a tool call or message turned down by a
  review hook, by the reviewer, or by moderation. Before, the rejection restarted
  the step and dropped the pending cancellation, so the agent finished replying
  to the first message and then answered the second one in a separate turn.
  Customers who send two messages in quick succession now get one reply that
  takes both into account.

### Security

- Raise the minimum `python-multipart` version to 0.0.30 for both uv
  and pip installations, preventing installation of versions affected by
  CVE-2026-53539.

## [4.0.1] - 2026-11-02

### Changed

- `parlant init --scaffold lead-gen` now creates its agent with
  `preamble_config=p.PreambleConfiguration.encourage()`. Before its first tool
  call the agent says one short line, such as "Let me check that", so the
  visitor isn't left waiting in silence. Without it, preambles were unmanaged:
  whatever the model wrote before a tool call went straight through.
- Every tool in the `lead-gen` and `retail-store` scaffolds now has a
  `narration`, such as "Checking which days are free" or "Tracking your
  shipment". The chat shows it while the tool runs, instead of the generic
  "Running tools".

### Fixed

- With the Anthropic provider, indexing no longer fails on Claude Opus 4.7 and
  later or any Claude 5 model, including the default routes (Sonnet 5.5 and
  Haiku 5.5). Parlant sent a `temperature`, which those models reject with
  "`temperature` is deprecated for this model."; it now sends one only to
  Claude 3 models and Claude 4 models up to 4.6, which still accept it.
- A reasoning effort on a Claude Fable model now uses adaptive thinking instead
  of a thinking budget, which Fable rejects.
- A reasoning effort on Claude Opus 4.5 no longer fails. Parlant sent it
  adaptive thinking, which it rejects ("adaptive thinking is not supported on
  this model"); it now gets a thinking budget, as Sonnet and Haiku 4.5 do.

## [4.0.0] - 2026-11-02

Parlant 4 release.
