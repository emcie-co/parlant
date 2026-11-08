# Queued responses within one process

MCN-453, under MCN-427. Design and implementation scope approved 2026-11-08.
Source base: develop `cbe3311cf3a3a8282b94c7646410c8293d993068`.

## Decision and limits

One process-local coordinator combines ready response requests, registered
background jobs and ordinary customer turns for a session. The user accepted
best-effort routing affinity, restart loss and possible competing replies when
traffic lands on separate pods. This supersedes the cross-pod/restart-safe
proposal discussed earlier on 2026-11-08.

Completed tool results and incurred costs retain their existing persisted
history/accounting path. The queue and coroutine registrations are memory-only.
There is no durable queue, lease, heartbeat/recovery protocol, transaction/outbox,
remote ownership, event-offset migration or automatic replay in this V1.

## SDK contract

```python
await agent.queue_response(session)

async def lookup_and_store() -> None:
    result = await lookup()
    await session.emit_tool_result(tool, result)

await agent.queue_response(session, after=lookup_and_store())
```

The await acknowledges scheduling, not generated text. Omitted `session` resolves
`Session.current`. An `after` coroutine is registered before launch and must save
results before returning. Register every job that should participate in a batch;
arbitrary tasks and not-yet-registered future jobs are invisible to coordination.

A failed job releases its registration and is logged. It does not invent an
answer or retry consequential work. Application code should persist an error
result if the customer needs an explanation. Other successful jobs proceed.
A failed generation releases that attempt's requests without an endless retry;
results and charges stay persisted for a later explicit request.

## What happens during an existing response?

1. A foreground response can satisfy queued requests only for inputs included
   in its snapshot. It does not wait for unfinished registered work.
2. New results before queued text is visible invalidate the unpublished reply
   and start a fresh generation. Completed jobs are never run again.
3. Once the registered batch settles, a foreground response that has not
   published text also rejects ready results missing from its snapshot; the
   queued successor reloads the conversation. While jobs remain pending, a
   foreground acknowledgment can proceed without clearing unseen results.
4. Once text is visible it stays. Uncovered results form one automatic successor,
   unless a new customer turn incorporates them first. Three calls do not produce
   three follow-ups, but an already-visible reply cannot retroactively include
   results that arrived later.
5. A newer customer message interrupts queued generation through normal session
   dispatch. Registered work survives cancellation of its originating turn.
6. If a customer turn already answered persisted results while their registered
   coroutine was finishing bookkeeping, its eventual completion does not create
   a redundant reply.

Background-only replies use `engine.utter`: responder-only, no fresh matching or
fresh tool calls. Customer messages use normal processing. Manual mode suppresses
queued text both at dispatch and publication. Explicit `Agent.utter()` remains
independent. API cancellation of an active queued reply cancels its local worker,
including its remaining pending jobs and requests.

## Implementation boundaries

- `core/response_coordination.py`: local state, batch registrations, input coverage,
  publication guards and service-owned worker lifecycle. Injected generation and
  history callbacks keep it independent of a concrete model or database.
- `core/app_modules/sessions.py`: ordinary dispatch begins/finishes a foreground
  run; queued rendering reuses independent utter tracing, accounting, diagnostics
  and cancellation statuses. Runtime-store selection follows engine emissions.
- `core/engines/compass/engine.py`: report the history actually loaded, including
  history represented by a compaction summary. Dispatch time is not coverage.
- `CoordinatedEventEmitter`: guard initial messages and every returned streaming
  updater at the persistence boundary. Empty placeholders are not visible text.
  Status events remain available for cancellation/completion listeners. Own
  engine tool results extend coverage; independent background emissions do not.
- `sdk.py`: scheduling/registration API, context inheritance and actionable limits.
  There is no HTTP queue endpoint or change to the cached responder prompt.

Customer-event identities and canonical tool-result fingerprints identify inputs.
The same session lock serializes input-intent updates and actual message writes;
there is no separate freshness check followed by an unguarded local write.
Incoming users pin state while waiting for that lock. An exhausted worker retires
without an intervening await; canceled workers unwind their jobs before releasing
state. Unstarted application coroutines are closed on registration failure or
pre-start cancellation.

### Edge-case corrections — 2026-11-08

The initial merge-readiness assessment was superseded after an edge audit.
Twenty-seven additional checks initially produced seven failures: four core and
three live SDK cases. Their assertions remain unchanged during the fixes.

- Reconcile request inputs with a fresh store view under the publication lock,
  rather than interpreting an older model snapshot as evidence of deletion.
  Retire removed requirements and keep still-relevant results. Validate before
  generation and first visible publication, without rereading full history for
  every subsequent streaming chunk.
- Concurrent registrations await the same worker-installation task. A worker
  canceled before entering its coroutine cannot own new application work;
  registration closes that unstarted work and releases queue state.
- Track copies of the chunks actually persisted by each message handle. Finalize
  an interrupted stream with that committed prefix plus its terminator, bypassing
  only the stale-text guard for this closing write. Do not flush the loop's live
  buffer, which may already contain an unpublished next chunk. Missing events
  after deletion require no replacement write; other closing failures are logged.
- Gemini generateContent cannot start history with an unpaired function response.
  Render leading external results as text context without inventing tool calls
  or signatures. Normal native call/result replay remains unchanged; the
  Interactions API retains its existing encoding.

The live regeneration regression originally stopped at a Gemini protocol error;
the adapter fix lets it reach and pass its queue assertions. A focused offline
encoding regression failed before that repair, while a native replay control
passed. The existing payload-packing test now supplies matching calls, keeping
its original response-payload assertions.

## Trace and cost attribution

Every generated queued reply has one fresh trace and one model-usage record,
using the existing session-operation export path. The latest pending request
when generation starts supplies the inherited originating-turn context. Later
requests already covered by that running reply do not change its trace origin.
Without an originating turn, the reply owns a new turn.

Each tool result retains its original trace and turn reference. A foreground
customer reply owns its normal response usage. Do not copy the combined model
record into each contributing job: one generated response is charged once.
There is no new multi-parent turn representation in the Dashboard. Native trace
links and `__parlant__.turn_trace_id` continue through existing export/ingestion.

## Tests and evidence

Tests and implementation gates were preapproved on 2026-11-08, including the
responder-only decision. Initial ten core cases failed against the unchanged
engine because the coordinator did not exist. The first real-model SDK baseline
failed on the missing `Agent.queue_response` API. Two later regressions reproduced
handoff-during-generation and completion-after-an-already-answering-turn before
those fixes were applied.

`tests/core/stable/test_response_coordination.py` covers local coalescing,
registered batches, foreground coverage, pre/post-publication changes,
interruption, streaming updater fencing, idle retirement, independent sessions,
job/generation failures and shutdown. Controlled async barriers and a recording
callback test timing without mocking the SDK or a model.

`tests/sdk/test_queued_responses.py` inherits `SDKTest` and exercises real HTTP,
engine and model paths. Semantic assertions grade relevant result coverage;
exact assertions check message counts, manual mode, cancellation, persisted
costs, current-session context and trace attribution. Existing session-emission
and independent-utter accounting tests protect compatibility.

Local verification on 2026-11-08 uses an isolated frozen all-extras environment,
not the shared checkout environment. No production connection or consumer
adoption is part of these tests. The workspace tracker and Linear contain actual
run results; a passing local run is not deployment evidence.
