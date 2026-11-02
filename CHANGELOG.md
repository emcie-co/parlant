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
