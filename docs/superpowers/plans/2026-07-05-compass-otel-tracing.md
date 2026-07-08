# Compass Otel Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument the Compass engine with the agreed OpenTelemetry operation spans and point-in-time events for rule matching, tool calling, review, loop output, effort raises, labels, compaction, and failures.

**Architecture:** Add a Compass-specific tracing helper that centralizes event names, domain-object-to-attribute mapping, and attribute serialization, then wire it through existing Compass phase boundaries. Use spans for lexical engine operations and typed tracer methods for facts observed inside those spans; do not mirror generic session status events into OTel.

**Tech Stack:** Python, Parlant `Tracer`, OpenTelemetry adapter via existing tracer abstraction, pytest.

---

## Review Findings

- The working tree is clean after the pull. Recent commits already renamed guidelines to rules/groups and added session rule and glossary pruning.
- Existing Compass spans are partial and use old or inconsistent names: `initialize`, `process`, `rule.recall`, `rule.rank`, `rule.distill`, `rule.prune`, `glossary.recall`, `glossary.prune`, `tools.run`, `sessions.compact.check`, and `sessions.compact.generate`.
- The proposed list has one duplicate: `match.tool.recall` appears twice. Keep one span.
- The proposed list has a spelling issue: `Glossasry` should be `Glossary`.
- The proposed `match.preload` description says "Preload and warm-up rule-matching inputs", but current code separates preload from warm-up. Keep `match.preload` for loading variables/rules/session rules; use `engine.process.warmup` and `engine.initialize` for cache warm-up.
- Streaming phases such as reasoning, message emission, and tool-call generation are represented by paired start/finish events under the active lexical span, typically `loop.step`. They are not represented as spans, because these phases cross stream event-handler boundaries and do not map to lexical operations.
- Compass call sites should pass domain objects to typed `CompassTracer` methods, following the alpha engine's `EngineTracer` pattern. Event-name and attribute construction should live in `src/parlant/core/engines/compass/tracing.py`, not inline in engine, matcher, loop, review, or tool-runner code.
- Generic session status emissions such as `typing`, `processing`, and `ready` are user-facing session events, not backend tracing events. Do not emit `loop.status`; emit explicit reasoning/message/tool telemetry instead.
- Boolean event names like `matched.rank.yes/no` and `compaction.checked.yes/no` should be implemented as concrete event names, for example `matched.rank.yes` and `matched.rank.no`, not with a slash in the name.
- `tool.requested` and `tool.called` are distinct. `tool.requested` happens when the model has produced final tool-call arguments. `tool.called` happens after review approval, immediately before external tool execution.

## Final Span Names

- `engine.initialize`
- `engine.process`
- `engine.process.warmup`
- `engine.process.finalize`
- `load.context`
- `load.variables`
- `match.preload`
- `match.fill`
- `match.update`
- `match.rule.recall`
- `match.rule.rank`
- `match.rule.distill`
- `match.rule.prune`
- `match.glossary.recall`
- `match.glossary.prune`
- `match.tool.recall`
- `loop.run`
- `loop.step`
- `tools.review`
- `tools.batch`
- `tools.call`
- `compaction.check`
- `compaction.compact`

## Final Event Names

- `loaded.variable`
- `loaded.glossary`
- `loaded.rule`
- `matched.function.yes`
- `matched.function.no`
- `matched.recall.yes`
- `matched.recall.no`
- `matched.rank.yes`
- `matched.rank.no`
- `matched.distill.yes`
- `matched.distill.no`
- `action.raise_effort`
- `action.add_label`
- `tool.requested`
- `tool.reviewed`
- `tool.called`
- `tool.result`
- `tool.error`
- `review.passed`
- `review.rejected`
- `loop.give_up`
- `loop.reasoning.started`
- `loop.reasoning`
- `loop.reasoning.finished`
- `loop.message.started`
- `loop.message.finished`
- `loop.tools.started`
- `loop.tools.finished`
- `loop.message`
- `loop.tool.transient`
- `loop.tool.persistent`
- `compaction.checked.yes`
- `compaction.checked.no`
- `compaction.compacted`
- `compaction.failed`
- `process.failed`

## File Structure

- Create `src/parlant/core/engines/compass/tracing.py`: Compass tracing helper, attribute serializers, and event helper methods.
- Modify `src/parlant/core/engines/compass/engine.py`: root spans, context loading span, finalization span, warm-up span, compaction events, process failure event.
- Modify `src/parlant/core/engines/compass/variable_loader.py`: `load.variables` span and `loaded.variable` events.
- Modify `src/parlant/core/engines/compass/matcher.py`: preload/fill/update spans, loaded rule events, effort raise event, session label persistence/event, and match-level wrapper spans.
- Modify `src/parlant/core/engines/compass/matching/rule_function_matcher.py`: rename span and emit per-rule function match yes/no events.
- Modify `src/parlant/core/engines/compass/matching/rule_recaller.py`: rename span and emit per-rule recall yes/no events.
- Modify `src/parlant/core/engines/compass/matching/rule_ranker.py`: rename span and emit per-rule rank yes/no events.
- Modify `src/parlant/core/engines/compass/matching/rule_distiller.py`: rename span and emit per-rule distill yes/no events.
- Modify `src/parlant/core/engines/compass/matching/rule_pruner.py`: rename span to `match.rule.prune`.
- Modify `src/parlant/core/engines/compass/matching/glossary_recaller.py`: rename spans and emit loaded glossary events.
- Modify `src/parlant/core/engines/compass/matching/tool_recaller.py`: add `match.tool.recall` span and include `candidate_count`, `scored_count`, and `selected_count` attributes on that span.
- Modify `src/parlant/core/engines/compass/loop/base_loop.py`: loop spans, phase start/finish events, reasoning/message/tool events, tool request/review/batch/call events, give-up event.
- Modify `src/parlant/core/engines/compass/loop/blocking_loop.py`: message phase start/finish and `loop.message` events for blocking output.
- Modify `src/parlant/core/engines/compass/loop/streaming_loop.py`: message phase start/finish and `loop.message` events for streaming output.
- Modify `src/parlant/core/engines/compass/tool_runner.py`: remove or rename the low-level `tools.run` span so `tools.call` is not double-counted.
- Modify `src/parlant/core/engines/compass/compacter.py`: rename compaction spans.
- Test `tests/core/stable/engines/compass/test_tracing.py`: tracing helper unit tests.
- Test existing Compass matcher/loop/tool/compacter tests with small assertions where the behavior is easiest to observe.

## Task 1: Add Compass Tracing Helper

**Files:**
- Create: `src/parlant/core/engines/compass/tracing.py`
- Test: `tests/core/stable/engines/compass/test_tracing.py`

- [ ] **Step 1: Write the failing helper tests**

```python
from contextlib import contextmanager
from typing import Mapping

from parlant.core.tracer import AttributeValue, Tracer
from parlant.core.engines.compass.tracing import CompassTracer, format_json_attr


class RecordingTracer(Tracer):
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict[str, AttributeValue]]] = []
        self.events: list[tuple[str, dict[str, AttributeValue]]] = []
        self._span_id = "<main>"

    @contextmanager
    def span(self, span_id: str, attributes: Mapping[str, AttributeValue] = {}):
        self.spans.append((span_id, dict(attributes)))
        previous = self._span_id
        self._span_id = span_id
        try:
            yield
        finally:
            self._span_id = previous

    @contextmanager
    def attributes(self, attributes: Mapping[str, AttributeValue]):
        yield

    @property
    def trace_id(self) -> str:
        return "trace-1"

    @property
    def span_id(self) -> str:
        return self._span_id

    def get_attribute(self, name: str) -> AttributeValue | None:
        return None

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        pass

    def add_event(self, name: str, attributes: Mapping[str, AttributeValue] = {}) -> None:
        self.events.append((name, dict(attributes)))

    def flush(self) -> None:
        pass


def test_format_json_attr_serializes_nested_values() -> None:
    assert format_json_attr({"x": 1, "nested": {"ok": True}}) == '{"x": 1, "nested": {"ok": true}}'


def test_compass_tracer_emits_events() -> None:
    tracer = RecordingTracer()
    compass = CompassTracer(tracer)

    compass.event("tool.called", {"tool_id": "svc:lookup", "arguments": {"x": 1}})

    assert tracer.events == [
        ("tool.called", {"tool_id": "svc:lookup", "arguments": '{"x": 1}'})
    ]

```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra gemini pytest tests/core/stable/engines/compass/test_tracing.py -q`

Expected: fail because `parlant.core.engines.compass.tracing` does not exist.

- [ ] **Step 3: Implement the tracing helper**

```python
# src/parlant/core/engines/compass/tracing.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import TypeGuard

from parlant.core.tracer import AttributeValue, Tracer


def format_json_attr(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_attribute_value(value: object) -> TypeGuard[AttributeValue]:
    return isinstance(value, (str, bool, int, float))


def _is_attribute_value_sequence(value: object) -> TypeGuard[AttributeValue]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return False
    if not value:
        return True
    return (
        all(type(v) is str for v in value)
        or all(type(v) is bool for v in value)
        or all(type(v) is int for v in value)
        or all(type(v) is float for v in value)
    )


def normalize_attrs(attributes: Mapping[str, object]) -> dict[str, AttributeValue]:
    normalized: dict[str, AttributeValue] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if _is_attribute_value(value) or _is_attribute_value_sequence(value):
            normalized[key] = value
        else:
            normalized[key] = format_json_attr(value)
    return normalized


class CompassTracer:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def event(self, name: str, attributes: Mapping[str, object] | None = None) -> None:
        self._tracer.add_event(name, normalize_attrs(attributes or {}))
```

- [ ] **Step 4: Run helper tests**

Run: `uv run --extra gemini pytest tests/core/stable/engines/compass/test_tracing.py -q`

Expected: pass.

## Task 2: Normalize Top-Level Engine and Compaction Spans

**Files:**
- Modify: `src/parlant/core/engines/compass/engine.py:85`
- Modify: `src/parlant/core/engines/compass/engine.py:113`
- Modify: `src/parlant/core/engines/compass/engine.py:166`
- Modify: `src/parlant/core/engines/compass/engine.py:290`
- Modify: `src/parlant/core/engines/compass/compacter.py:91`
- Modify: `src/parlant/core/engines/compass/compacter.py:113`

- [ ] **Step 1: Rename spans**

Replace:

```python
with self._tracer.span("initialize", ...):
```

with:

```python
with self._tracer.span("engine.initialize", ...):
```

Replace:

```python
with self._tracer.span("process", ...):
```

with:

```python
with self._tracer.span("engine.process", ...):
```

Wrap the `finalize_turn` body:

```python
with self._tracer.span("engine.process.finalize"):
    await self._refresh_interaction_history(engine_context)
    await self._compact_if_needed(engine_context)
    await self._matcher.prune_session_rules(engine_context)
    await self._matcher.prune_session_glossary(engine_context)
    with self._tracer.span("engine.process.warmup"):
        await self._matcher.warm_up(engine_context)
    await self._update_usage(engine_context)
```

Replace compacter spans:

```python
with self._tracer.span("compaction.check"):
```

and:

```python
with self._tracer.span("compaction.compact"):
```

- [ ] **Step 2: Add compaction and failure events**

In `_compact_if_needed`, emit `compaction.checked.yes` or `compaction.checked.no` based on `needs_compaction`. Emit `compaction.compacted` after summary persistence. Emit `compaction.failed` in the exception handler.

In the `process()` exception handler, before returning `False`, emit through the typed tracer:

```python
CompassTracer(self._tracer).process_failed(e)
```

- [ ] **Step 3: Run focused compacter tests**

Run: `uv run --extra gemini pytest tests/core/stable/engines/compass/test_compacter.py -q`

Expected: pass.

## Task 3: Instrument Context Loading and Matcher Phases

**Files:**
- Modify: `src/parlant/core/engines/compass/engine.py:175`
- Modify: `src/parlant/core/engines/compass/variable_loader.py:51`
- Modify: `src/parlant/core/engines/compass/matcher.py:139`
- Modify: `src/parlant/core/engines/compass/matcher.py:151`
- Modify: `src/parlant/core/engines/compass/matcher.py:162`
- Modify: `src/parlant/core/engines/compass/matcher.py:769`

- [ ] **Step 1: Add spans**

Wrap `_load_context` body with `load.context`. Wrap `VariableLoader.load()` with `load.variables`. Wrap `Matcher.preload()`, `Matcher.fill()`, and `Matcher.update()` with `match.preload`, `match.fill`, and `match.update`.

- [ ] **Step 2: Emit load events**

In `VariableLoader.load()`, after loaded values are computed:

```python
CompassTracer(context.tracer).context_variables_loaded(loaded)
```

Emit the value itself so the UI can resolve and display the loaded context variable
entity. `CompassTracer` normalizes non-primitive values to JSON attributes.

In `Matcher.preload()`, after `usable_rules` is set:

```python
CompassTracer(context.tracer).rules_loaded(context.state.usable_rules)
```

- [ ] **Step 3: Emit effort and label events from `_record()`**

Capture old effort before mutating matches, then compare after `invalidate_cached_properties()`.

```python
old_effort = context.state.dynamic_effort_level
...
context.state.invalidate_cached_properties()
new_effort = context.state.dynamic_effort_level
if new_effort > old_effort:
    CompassTracer(context.tracer).effort_raised(
        old_effort,
        new_effort,
        [
            str(match.rule.id)
            for match, usage in matches
            if usage == _ContextUsage.MATCH_CURRENT_TURN
            and match.rule.effort == new_effort
        ],
    )
```

Add a private `_update_session_labels()` called from `_record()` after match state is updated. Persist only labels not already present on `context.session.labels`; emit `action.add_label` only after `upsert_session_labels()` succeeds.

- [ ] **Step 4: Run matcher tests**

Run: `uv run --extra gemini pytest tests/core/stable/engines/compass/matching/test_matcher.py -q`

Expected: pass.

## Task 4: Rename and Enrich Matching Component Spans and Events

**Files:**
- Modify: `src/parlant/core/engines/compass/matching/rule_function_matcher.py`
- Modify: `src/parlant/core/engines/compass/matching/rule_recaller.py`
- Modify: `src/parlant/core/engines/compass/matching/rule_ranker.py`
- Modify: `src/parlant/core/engines/compass/matching/rule_distiller.py`
- Modify: `src/parlant/core/engines/compass/matching/rule_pruner.py`
- Modify: `src/parlant/core/engines/compass/matching/glossary_recaller.py`
- Modify: `src/parlant/core/engines/compass/matching/tool_recaller.py`

- [ ] **Step 1: Rename spans**

Use these replacements:

```text
rule.function_match -> match.rule.function
rule.recall         -> match.rule.recall
rule.rank           -> match.rule.rank
rule.distill        -> match.rule.distill
rule.prune          -> match.rule.prune
glossary.recall     -> match.glossary.recall
glossary.prune      -> match.glossary.prune
```

Add `match.tool.recall` around `ToolRecaller.prepare()`.

- [ ] **Step 2: Emit match result events**

For every result object returned by each matcher, emit the suffix event based on `is_relevant`.

```python
CompassTracer(context.tracer).rules_ranked(ranked_rules)
```

Apply the same pattern for function, recall, and distill results using event names `matched.function.*`, `matched.recall.*`, and `matched.distill.*`.

- [ ] **Step 3: Emit glossary events**

After `GlossaryRecaller.recall()` sets `context.state.glossary_terms`, emit `loaded.glossary` per term with `term_id`, `name`, and `last_modified`.

- [ ] **Step 4: Run matching component tests**

Run:

```bash
uv run --extra gemini pytest \
  tests/core/stable/engines/compass/matching/test_rule_function_matcher.py \
  tests/core/stable/engines/compass/matching/test_rule_recaller.py \
  tests/core/stable/engines/compass/matching/test_rule_ranker.py \
  tests/core/stable/engines/compass/matching/test_rule_distiller.py \
  tests/core/stable/engines/compass/matching/test_rule_pruner.py \
  tests/core/stable/engines/compass/matching/test_glossary_recaller.py \
  tests/core/stable/engines/compass/matching/test_tool_recaller.py \
  -q
```

Expected: pass.

## Task 5: Instrument Loop, Reasoning, Message, Review, and Tool Calls

**Files:**
- Modify: `src/parlant/core/engines/compass/loop/base_loop.py`
- Modify: `src/parlant/core/engines/compass/loop/blocking_loop.py`
- Modify: `src/parlant/core/engines/compass/loop/streaming_loop.py`
- Modify: `src/parlant/core/engines/compass/tool_runner.py`

- [ ] **Step 1: Add loop spans**

Wrap `BaseLoop.run()` body with `loop.run`. Wrap `_run_step()` body with `loop.step`, with attributes `step_index=len(state.steps) + 1` and `disable_tools=state.disable_tools`.

- [ ] **Step 2: Emit loop output events**

Do not mirror generic session status events. Emit `loop.reasoning` when reasoning is finalized, `loop.message` when an assistant message is emitted, and `loop.tool.transient` / `loop.tool.persistent` when tool result events are emitted. Build these payloads through typed `CompassTracer` methods.

- [ ] **Step 3: Emit phase start/finish events**

Emit paired phase events under the active `loop.step` span:

- `loop.reasoning.started` / `loop.reasoning.finished`
- `loop.message.started` / `loop.message.finished`
- `loop.tools.started` / `loop.tools.finished`

Track phase state in `_LoopState` so events are idempotent and so unusual stream exits can close any open phases in `_run_step()` cleanup. Do not add manual span lifecycle helpers for these phases.

- [ ] **Step 4: Emit review events**

Wrap `review_tool_calls()` with `tools.review` only when review actually runs. If review returns no adjusted reasoning, emit `review.passed`; otherwise emit `review.rejected` with `has_adjusted_reasoning`, `has_todo`, and redacted `todo`/`adjusted_reasoning` only if a debug flag is later added.

Emit `tool.reviewed` per tool call participating in the review with `tool_call_id`, `tool_name`, `consequential`, and `approved`.

- [ ] **Step 5: Emit tool request and execution events**

After `StepCompleted(result.needs_tools=True)` and before review:

```python
CompassTracer(context.tracer).tool_calls_requested(result.tool_calls)
```

Wrap `run_tool_calls()` with `tools.batch`. In `run_tool_call()`, wrap the external execution with `tools.call`; emit `tool.called` before calling the runner, and emit `tool.result` or `tool.error` after the result is known.

- [ ] **Step 6: Remove duplicate low-level tool span**

In `ToolRunner.run_tool()`, remove `with self._tracer.span("tools.run")` so individual execution is timed once by `tools.call` in the loop controller.

- [ ] **Step 7: Run loop and tool tests**

Run:

```bash
uv run --extra gemini pytest \
  tests/core/stable/engines/compass/loop/test_blocking_loop.py \
  tests/core/stable/engines/compass/loop/test_streaming_loop.py \
  tests/core/stable/engines/compass/test_tool_runner.py \
  tests/core/stable/engines/compass/test_reviewer.py \
  -q
```

Expected: pass.

## Task 6: Verify Full Compass Behavior

**Files:**
- No source files beyond previous tasks.

- [ ] **Step 1: Run stable Compass tests**

Run:

```bash
uv run --extra gemini pytest tests/core/stable/engines/compass -q
```

Expected: pass.

- [ ] **Step 2: Run SDK tests that exercise Compass user flows**

Run:

```bash
uv run --extra gemini pytest tests/sdk/test_rules.py tests/sdk/test_retrievers.py tests/sdk/test_agents.py -q
```

Expected: pass.

- [ ] **Step 3: Search for old span/event names**

Run:

```bash
rg -n '"initialize"|"process"|"rule\\.recall"|"rule\\.rank"|"rule\\.distill"|"rule\\.prune"|"glossary\\.recall"|"glossary\\.prune"|"tools\\.run"|"sessions\\.compact' src/parlant/core/engines/compass
```

Expected: no remaining Compass span names from the old taxonomy.

- [ ] **Step 4: Commit**

```bash
git add \
  src/parlant/core/engines/compass \
  tests/core/stable/engines/compass \
  docs/superpowers/plans/2026-07-05-compass-otel-tracing.md
git commit -m "feat: instrument compass otel tracing"
```

Expected: commit succeeds after tests pass.

## Self-Review

- Spec coverage: the plan covers every proposed span and event, including tool arguments, effort raising, labels, review pass/reject, reasoning/message/tool emitted events, compaction, and process failure.
- Placeholder scan: the plan avoids unresolved placeholders and uses concrete file paths, event names, and test commands.
- Type consistency: helper code uses existing `Tracer`, `AttributeValue`, and Engine domain-object conventions. Streaming phase timing uses paired events under `loop.step`, avoiding manual span lifecycle helpers.
