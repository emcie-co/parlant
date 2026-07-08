# Usage Reporting & Cost Visibility

This document explains how Parlant accounts for model usage, where to see it,
and how to set up advisory cost thresholds — including plugging in your own
cost-control policy.

## The model

Three pieces, each with one job:

| Question | Component | Default |
|---|---|---|
| **How many tokens did we spend?** | `UsageReporter` | Always on — pure accounting, per turn, per model |
| **Should this work be allowed / flagged?** | `CostControlPolicy` (port) | `AdvisoryCostControlPolicy` — accounts and warns, never blocks |
| **What does a token *cost*?** | `CostWeights` | Weighted units: uncached input ×1.0, cached ×0.1, output ×4.0 |

Every provider call — response generation, matching, embeddings — funnels its
observed usage through `UsageReporter.report_usage`. Cost policies observe that
stream; they never sit inside provider adapters.

## Where usage shows up

### Session metadata

At the end of every turn, the engine writes the turn's aggregated usage into
the session's metadata under the `"usage"` key, keyed by trace (one trace per
turn), then by model:

```json
{
  "usage": {
    "<trace_id>": {
      "gpt-5.4-nano": {
        "input_tokens": 48210,
        "output_tokens": 512,
        "cached_input_tokens": 45800,
        "extra": {"reasoning_tokens": 128}
      }
    }
  }
}
```

Read it via the sessions API (`GET /sessions/{id}`) like any other metadata.
Note the `cached_input_tokens` field — see [Weighted cost units](#weighted-cost-units)
for why it matters.

### Programmatic observation

`UsageReporter.add_listener` registers a callback invoked on every report:

```python
container[UsageReporter].add_listener(
    lambda trace_id, model, usage: my_metrics.increment(model, usage.input_tokens)
)
```

Listener contract: exceptions are swallowed (accounting must never be affected
by an observer), and callbacks run on the hot path — keep them fast, queue
anything heavy.

## Weighted cost units

Thresholds in Parlant are defined over **weighted units**, not raw tokens.
The engine is deliberately cache-heavy: a healthy session re-reads a large
cached prompt prefix many times per turn, and cached input is roughly an order
of magnitude cheaper than uncached. Raw token counts would flag exactly the
sessions the architecture makes cheap.

`CostWeights` converts a usage report into units:

```
units = uncached_input × 1.0  +  cached_input × 0.1  +  output × 4.0
```

(`input_tokens` is reported inclusive of `cached_input_tokens`; the weighting
handles the split.) Per-model multipliers let you make a large model's tokens
count more than a nano model's:

```python
CostWeights(model_multipliers={"big-model": 10.0})
```

Rule of thumb: think of a unit as "one uncached input token equivalent" —
100k cached input ≈ 10k units, 1k output ≈ 4k units.

## Advisory thresholds

The built-in policy keeps a **decaying per-session window** of weighted units
(default span: 5 minutes) and, when a configured threshold is crossed, attaches
warnings that the engine logs — while always allowing the work. No threshold is
configured by default, so out of the box you get accounting with no noise.

Enable warnings by rebinding the policy with your numbers — via
`configure_container` in the SDK:

```python
import parlant.sdk as p
from lagom import Container
from parlant.core.cost_control import AdvisoryCostControlPolicy, CostControlPolicy
from parlant.core.usage_reporter import UsageReporter

async def configure_container(container: Container) -> Container:
    container[CostControlPolicy] = AdvisoryCostControlPolicy(
        container[UsageReporter],          # must be the container's instance
        advisory_threshold_units=250_000,  # warn past ~250k units...
        window_seconds=300,                # ...within any 5-minute window
    )
    return container

async with p.Server(configure_container=configure_container) as server:
    ...
```

or the equivalent in a server module (`parlant-server run --module my_module`),
whose `configure_module(container)` does the same rebinding.

When a session runs hot you'll see log warnings like:

```
Cost-control warning: Session <id> crossed the advisory cost threshold:
312450 weighted units in the last 300s (threshold: 250000)
```

## The cost-control policy port

Everything above rides one small interface, which you can implement yourself:

```python
class CostControlPolicy(ABC):
    async def check(self, context: CostContext, work: WorkKind) -> CostVerdict: ...
    def report(self, trace_id: str, model: str, usage: UsageInfo) -> None: ...
```

- `check` is called by the engine at three coarse choke points — `TURN` (before
  a turn's preparation begins), `STEP` (each response-loop step boundary), and
  `BACKGROUND` (post-response cache warm-ups and pruning) — with the full
  identity of the work (`agent_id`, `session_id`, `customer_id`, `trace_id`).
- `report` receives every observed usage event (the policy self-subscribes to
  the `UsageReporter` at construction).
- The verdict is richer than a boolean: `allowed`, `warnings`,
  `retry_after_utc`, `reason`, `scope`.

Contracts your implementation must honor (and may rely on):

- **`check` fails open.** If it raises, the engine logs and proceeds.
- **`report` must not block.** It runs on the hot path; queue internally if you
  persist state.
- **Trace binding**: `check` calls carry the context; usage reported for traces
  you've never seen should be accounted to an "unattributed" bucket, not
  dropped.

A policy **may deny work** — the engine honors it at all three choke points,
including a client-visible cooldown protocol for denied turns. The built-in
policy never does; for the enforcing implementation that ships with Parlant,
see [Pro cost controls](pro/cost_control.md). If you build your own windowed
policy, extend `WindowedCostControlPolicy`, which provides the accounting base
(weighted windows, trace binding, the unattributed bucket) so you only write
the verdict logic.
