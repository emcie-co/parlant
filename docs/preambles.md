# Tool Preambles

## Overview

A preamble is a short message that Parlant may produce immediately before it
calls a tool, such as “Let me check that for you.” It lets the user know that
work is happening before a potentially slow operation.

Configure them per agent with `preamble_config`:

```python
import parlant.sdk as p


agent = await server.create_agent(
    id="my-agent",
    name="Support Agent",
    instructions="Help customers with their orders.",
    preamble_config=p.PreambleConfiguration.encourage(),
)
```

## Motivation

Tool calls can take long enough that silence feels like a stalled conversation,
especially in voice and streaming channels. A concise preamble provides early
feedback while the tool runs.

That feedback is not desirable in every deployment. Fast tools may not need it,
and some voice experiences sound repetitive if every tool call begins with an
update. Parlant therefore lets you leave preambles unmanaged, encourage them at
a controlled interval, or suppress them.

## Tutorial and options

### Leave preambles unmanaged

```python
preamble_config=p.PreambleConfiguration.default()
# Or simply don't configure the parameter...
```

`default()` adds no preamble-specific prompt guidance and applies no special
filtering or trimming. The model may still produce text before a tool call; if it
does, Parlant passes that text through normally.

This is also the default for block and stream agents when `preamble_config` is omitted.

### Encourage preambles

```python
preamble_config=p.PreambleConfiguration.encourage()
```

`encourage()` asks the model for one short, natural sentence before the first
tool call. It also limits repeated progress updates and trims an emitted
preamble to one sentence.

Encouragement is model guidance, not a guarantee that a preamble will be
generated. The interval controls whether later pre-tool updates may be shown; it
does not schedule messages on a timer.

The default interval is 15 seconds. To change it:

```python
preamble_config=p.PreambleConfiguration.encourage(interval_seconds=8)
```

Voice workers use `encourage()` when an agent has no explicit preamble
configuration. Pass `default()` explicitly if you want a voice agent to remain
unmanaged.

### Discourage preambles

```python
preamble_config=p.PreambleConfiguration.discourage()
```

`discourage()` tells the model to call tools silently. Parlant also enforces the
choice at runtime: text generated immediately before a tool call is not
delivered to the user.

For streaming and voice, Parlant must briefly buffer possible preamble text
until it knows whether a tool call follows. If no tool call follows, that text is
an ordinary response and is delivered normally.

### Choosing an option

| Option | Prompt guidance | Runtime behavior | Typical use |
|---|---|---|---|
| `default()` | None | Pass pre-tool text through | Let the model behave naturally |
| `encourage()` | Ask for concise, timely progress updates | Apply interval gating and one-sentence trimming | Slow tools, streaming, and voice |
| `discourage()` | Ask for silent tool use | Suppress text immediately before a tool call | Fast tools or tightly controlled voice UX |

## Advanced usage

### Preamble decisions

Before each step that may call a tool, Parlant computes a **preamble decision**
— its judgment of whether this is a good moment for a progress update,
based on what the customer has already seen this turn:

- `ALLOW_INITIAL` — nothing user-visible has been shown yet this turn; a
  first "let me check that" message is helpful here.
- `ALLOW_INTERVAL_UPDATE` — something was shown, but more than the configured
  interval has passed since the last user-visible message; the customer may
  be wondering what's happening.
- `SUPPRESS` — the customer heard from the agent recently (within the
  interval); another update would be noise.

The decision is timing state, not configuration — you never set it. Parlant
derives it at each step from the turn's emission history and your
`interval_seconds`. The decision is then used in two places. The
**note factory** receives it, so the prompt guidance can fit the moment: ask for a
first preamble, or request silence. Under the `INTERVAL` emission policy, it
also controls at runtime whether pre-tool text is actually delivered. The
built-in `encourage()` behavior is exactly this: a tailored note per
decision, plus interval gating.

### Define a custom policy

Construct `PreambleConfiguration` directly when the standard options do not fit.
A note factory receives the current engine context, the timing decision, and the
configured interval. It returns prompt guidance for that step, or `None`:

```python
import parlant.sdk as p


def preamble_note(
    context: p.EngineContext,
    decision: p.PreambleDecision,
    interval_seconds: float,
) -> str | None:
    if decision is p.PreambleDecision.SUPPRESS:
        return "Run the next tool silently; the user was updated recently."

    return "Before using a tool, briefly state which account detail you are checking."


custom_preambles = p.PreambleConfiguration(
    interval_seconds=10,
    note_factory=preamble_note,
    emission_policy=p.PreambleEmissionPolicy.INTERVAL,
)
```

The emission policy controls runtime handling independently of the note:

- `PASSTHROUGH` always permits pre-tool text.
- `INTERVAL` permits an initial preamble and later updates after the configured
  interval, with one-sentence trimming.
- `SUPPRESS` never delivers pre-tool text.

### Set a deployment-wide policy

`PreambleConfigurationProvider` is the runtime source of per-agent settings.
Replace it with `configure_container` to apply a deployment default while still
allowing explicit `preamble_config` values to win:

```python
import parlant.sdk as p


class DeploymentPreambles(p.PreambleConfigurationProvider):
    def get(self, agent_id: p.AgentId) -> p.PreambleConfiguration | None:
        return super().get(agent_id) or p.PreambleConfiguration.discourage()


async def configure_container(container: p.Container) -> p.Container:
    container[p.PreambleConfigurationProvider] = DeploymentPreambles()
    return container


async with p.Server(configure_container=configure_container) as server:
    ...
```

The built-in provider stores configuration in process memory. Reapply SDK or
container configuration at startup; it is not persisted with the agent record.
This also lets custom configurations use Python callbacks such as
`note_factory`, which are not serializable.
