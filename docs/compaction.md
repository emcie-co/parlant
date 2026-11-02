# Session Compaction

Long-running sessions accumulate history without limit. Compaction is how
Parlant handles that: when a session's interaction history grows past a
size threshold, Parlant summarizes the earlier part of the conversation
into a single durable summary and continues the session on top of it. From
that point on, every prompt Parlant builds — responding, matching,
reviewing, pruning — sees the summary plus only the events that came after
it, instead of the entire raw history.

Compaction is on by default and needs no setup. This guide explains what it
does, how to tune it, and how it works underneath — enough to predict when a
session will compact, what survives the compaction, and how to debug it.

## Motivation

Three pressures build up as a session grows, and compaction relieves all of
them:

- **Cost and latency.** Every model call a turn makes — the responder, the
  per-rule ranking fan-out, reviews, discovery queries — carries the
  interaction history, directly or indirectly. Token cost therefore grows
  linearly with session length, on every call, on every turn. Past a point
  you are paying for the same stale transcript hundreds of times over.
- **Response quality.** Models attend less reliably over very long
  histories; details buried in the middle of a huge transcript get missed. A
  deliberate summary, written with the agent's policies in view, preserves
  the load-bearing facts in a form the model actually attends to.
- **Context limits.** Without compaction, a long enough session eventually
  overflows the model's context window and fails outright.

The summary is written with care: it is generated with explicit
instructions to preserve exactly the material a future turn needs — facts,
commitments, tool results, unresolved tasks, policy-sensitive state — and to
distinguish what was confirmed from what was assumed.

## Key terms

**Compaction marker**
: A SYSTEM message event persisted in the session's event stream, carrying
  the summary text and
  `metadata={"__parlant__": {"schema_version": 1, "source": "compacter"}}`.
  It is both the
  durable record of the summary and the point that splits "before" from
  "after": when a turn loads the session, everything before the last marker
  is replaced by the marker's summary.

**Session summary**
: The text of the most recent compaction marker, loaded into
  `state.session_summary` each turn and rendered into every component's
  prompt alongside the post-marker events.

**Compaction policy**
: A per-effort pair of `token_threshold` (when to compact) and
  `detail_level` (how much the summary preserves). Parlant picks the
  policy for the turn's dynamic effort level (see [Adaptive Effort](adaptive-effort.md)).

**Detail level**
: `LOW`, `MEDIUM`, or `HIGH` — how aggressively the summary compresses.
  See "Detail levels" below.

## When compaction triggers

Compaction runs **after** the turn's response has already been sent, as part
of the engine's finalize phase. The user never waits on it.

At finalize, Parlant estimates the token count of the interaction history
(the events since the last compaction, including the reply it just sent) and
compares it against the policy's `token_threshold` for the turn's effort
level. The defaults:

- `MIN` / `LOW` effort: compact at ~24,000 tokens, `LOW` detail.
- `MEDIUM` effort: compact at ~32,000 tokens, `MEDIUM` detail.
- `HIGH` effort: compact at ~48,000 tokens, `HIGH` detail.
- `MAX` effort: compact at ~64,000 tokens, `HIGH` detail.

Higher effort tolerates a longer raw history before compacting, and
preserves more when it does. This matches what effort means in general:
higher effort spends more to lose less information.

Note that the effort consulted is the *dynamic* effort level, so a matched
rule's triggered effort can change which policy applies on a given turn.

What you observe in the session when it triggers:

1. A status event: `Compacting session`.
2. A SYSTEM message event authored by "System" containing the summary, with
   `metadata.__parlant__.source == "compacter"` — the compaction marker.
   Front-ends that don't want to display summaries can filter on that
   metadata.

Compaction is best-effort by design: if the summary generation fails, the
error is logged and the turn is unaffected — the response already went out.
The history is still over the threshold, so Parlant tries again at the
end of the next turn.

## Detail levels

The detail level shapes the summarization instructions and how much
reasoning the model spends on them:

- **LOW** — compact aggressively. Keeps only durable facts, active user
  goals, important tool results, unresolved tasks, and policy-sensitive
  state. Minimal reasoning effort.
- **MEDIUM** — everything LOW keeps, plus major decision points,
  confirmations, denials, and enough chronology to explain how the session
  reached its current state. Low reasoning effort.
- **HIGH** — precise facts, identifiers, preferences, confirmations,
  denials, summarized tool calls and results, unresolved ambiguities,
  policy-sensitive details, and the latest state of every active request.
  Medium reasoning effort.

Whatever the level, the summarizer is instructed never to invent facts,
never to include private reasoning, and to write a summary that supersedes
any previous one.

## Configuring compaction

The policy table lives on the `Compacter` component and is replaced
programmatically via the SDK's container hooks. `set_policy` takes a
complete mapping — one entry per effort level — and validates it (missing
effort levels or non-positive thresholds raise immediately):

```python
import parlant.sdk as p
from parlant.core.engines.compass.compacter import (
    Compacter,
    CompactionDetail,
    CompactionPolicy,
)


async def initialize(container: p.Container) -> None:
    container[Compacter].set_policy(
        {
            p.Effort.MIN: CompactionPolicy(
                token_threshold=16_000, detail_level=CompactionDetail.LOW
            ),
            p.Effort.LOW: CompactionPolicy(
                token_threshold=16_000, detail_level=CompactionDetail.LOW
            ),
            p.Effort.MEDIUM: CompactionPolicy(
                token_threshold=24_000, detail_level=CompactionDetail.MEDIUM
            ),
            p.Effort.HIGH: CompactionPolicy(
                token_threshold=32_000, detail_level=CompactionDetail.HIGH
            ),
            p.Effort.MAX: CompactionPolicy(
                token_threshold=48_000, detail_level=CompactionDetail.HIGH
            ),
        }
    )


async def main() -> None:
    async with p.Server(initialize_container=initialize) as server:
        ...
```

Tuning guidance:

- **Lower thresholds** compact earlier: cheaper long sessions, but more
  frequent summarization calls and earlier loss of verbatim history.
- **Higher thresholds** keep more raw history: better fidelity for
  reference-heavy conversations, at linearly growing per-turn cost.
- **Detail level** is the fidelity/cost knob *within* a compaction. If your
  sessions carry lots of identifiers and in-flight requests (order numbers,
  claim states), prefer `HIGH` detail even at lower efforts.

The summarization model itself is selected through model routing under the
`"compaction"` route (the token estimate uses the `"tokenizer"` route), so
it can be pointed at a different model than the responder via the model
configuration service, like any other route.

## How it works under the hood

### Loading: the marker splits the session

When a turn begins, Parlant builds its working interaction by scanning
the session's events in order. Each time it encounters a compaction marker,
it discards everything accumulated so far and loads the marker's summary
into `state.session_summary`. The result: the working interaction is always
"the latest summary + every event after the latest marker". The marker
event itself is not part of the interaction — it is a boundary, not a
message.

Because the marker is a persisted session event, this is fully
deterministic across restarts and across processes: any engine instance
loading the session reconstructs the same compacted view.

### During the turn: everyone sees summary + suffix

The summary is not just for the responder. Each component renders it in its
own way, always alongside the post-marker events:

- The **response loop** injects a system message ("The earlier part of this
  session was compacted into the following summary…") between the system
  instructions and the chat history.
- The **reviewer**, **rule ranker**, and **rule pruner** each render a
  Session Summary section in their prompts.
- The **matcher's discovery queries** (rule recall, tool recall, glossary
  discovery) prepend `Session summary: …` to the query text they embed — so
  recall keyed off something discussed before the compaction still works.

### Triggering: the finalize sequence

After the response is emitted, Parlant runs a finalize sequence that is
shielded from cancellation, in this order:

1. **Refresh** the interaction from the store, so the check sees the reply
   that was just sent.
2. **Check**: format the interaction as JSON event lines (status events
   excluded), estimate tokens with the routed tokenizer, compare with the
   policy threshold. The check logs its numbers at debug level
   (`Compaction check: effort=…, tokens=…, threshold=…, needed=…`).
3. **Compact** if needed (details below), update `state.session_summary`
   in-process, and persist the marker event.
4. **Prune** the discovered-rule and glossary working sets — deliberately
   *after* compaction, so the pruner judges staleness against the fresh
   summary rather than the raw history.
5. **Warm up** next turn's prompt caches — last, so the prefixes it
   pre-fills reflect the compacted state.

Steps 4–5 sit behind the background cost gate, but compaction
itself is exempt from it: compacting *reduces* future cost, so blocking it
during a cost event would be self-defeating.

This ordering is why compaction never costs the user anything. The whole
sequence happens between turns. Compaction also changes the shape of every
prompt prefix, which causes expensive cache misses — but the warm-up absorbs
those before the next customer message arrives.

### The compaction prompt

The summary is produced by a single schematic generation on the
`"compaction"` route. The prompt contains, in order:

1. **Task description** — summarize so a future turn can continue
   correctly; no user-facing response; no invented facts; no hidden
   reasoning; preserve facts, requests, commitments, policy-sensitive
   details, tool calls and results, confirmations, denials, unresolved
   questions, missing information, remaining tasks; distinguish confirmed
   facts from assumptions; supersede any previous summary.
2. **Output format** — a JSON object with a single `summary` field.
3. **Detail-level instructions** — per the policy's `detail_level`.
4. **The agent's full system instructions** — identity, customer, context
   variables, glossary, and the complete rule inventory (including
   statically-included cross-referential rules). This is what makes the
   summary *policy-aware*: the summarizer knows which details the agent is
   obligated to act on, so it knows what it must not drop.
5. **The previous session summary**, if one exists — compactions chain, and
   each new summary subsumes the last.
6. **The full interaction history** as JSON events.

The generation carries a per-session prompt-cache hint (keyed
`<session_id>.compacter`, with the cache breakpoint at the
interaction-history header), so repeated compactions of the same session
reuse the cached instruction prefix and only pay for the history tail.

### Chained compactions

Nothing special happens on the second, third, or Nth compaction: the
summary-so-far is part of the compaction prompt, the model folds it into
the new summary, and the new marker becomes the session's split point. Old
markers and pre-compaction events remain in the session store untouched —
compaction never deletes history; it only changes what Parlant loads
into its working view.

## Debugging

- **"Why hasn't this session compacted?"** — Check the debug log for
  `Compaction check:` lines; they print the estimated token count, the
  active threshold, and the effort level that selected it. Remember the
  check runs post-response, so a session crossing the threshold mid-turn
  compacts at the end of that turn, not during it.
- **"Why did the agent forget X after compaction?"** — Find the marker
  event in the session (SYSTEM message,
  `metadata.__parlant__.source == "compacter"`)
  and read the summary: X either didn't survive summarization (consider a
  higher `detail_level` or a higher threshold) or was never in the history
  to begin with. The full compacted prompt is available at trace level
  (`Compacter prompt:`).
- **"Compaction failed"** — Look for `Session compaction failed after response generation`
  in the error log; the turn itself is unaffected and
  Parlant retries at the end of the next turn.
