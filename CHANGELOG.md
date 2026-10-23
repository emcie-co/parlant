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

### Changed

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
