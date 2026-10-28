# Session and Event Metadata

## Synopsis

Application metadata remains at the top level of session and event metadata.
Parlant-owned persisted annotations live under the reserved `__parlant__`
object. Compass stages its internal session changes in memory during a turn and
commits them together with compare-and-swap protection when processing
finishes.

## Ownership

A fully populated session has this general shape:

```json
{
  "application_key": "owned by the application",
  "__parlant__": {
    "schema_version": 1,
    "auth": {
      "guest_instance_id": "guest-instance-id"
    },
    "compass": {
      "input_checkpoint": {
        "event_id": "event-id",
        "offset": 12,
        "modified_utc": "2026-07-28T10:29:59+00:00",
        "content_sha256": "hex-encoded-sha256"
      },
      "needs_rebuild": false,
      "discovered_rule_ids": ["rule-id"],
      "evicted_discovered_rules": {
        "old-rule-id": 8
      },
      "last_discovered_rule_pruning_offset": 8,
      "session_terms": {
        "term-id": 12
      }
    },
    "usage": {
      "records": {},
      "baselines": {}
    },
    "voice": {
      "channel": "room-id",
      "participant": "participant-id"
    }
  }
}
```

An event can contain application annotations beside event-kind-specific
Parlant annotations:

```json
{
  "application_annotation": "owned by the application",
  "__parlant__": {
    "schema_version": 1,
    "diagnostic_report_uri": "/debug/reports/trace-id",
    "source": "compacter",
    "reasoning": true,
    "provider_data": {},
    "voice": {
      "speculative": false,
      "spoken_text": "The words actually played"
    }
  }
}
```

This event example is a union for reference. Most event kinds contain only one
or two of those internal fields.

## Using application metadata

Create sessions and events with ordinary top-level keys:

```python
session = await client.sessions.create(
    agent_id=agent_id,
    metadata={"account_tier": "gold"},
)

await client.sessions.create_event(
    session_id=session.id,
    kind="message",
    source="customer",
    message="Hello",
    metadata={"channel": "web"},
)
```

The update APIs use top-level set and unset operations. They preserve both
unmentioned application keys and the internal namespace:

```python
await client.sessions.update(
    session_id=session.id,
    metadata={
        "set": {"account_tier": "platinum"},
        "unset": ["temporary_flag"],
    },
)
```

`__parlant__` is readable because clients and the integrated UI consume some
internal annotations, but it is not application-writable. Session creation,
event creation, API/SDK updates, and tunnel updates reject attempts to set or
unset it. Do not use names below this object as application storage.

Note the distinct-but-similarly-named `metadata` on a *tool result*
(`result.metadata`, nested inside a tool event). It is a separate, frontend-only
channel — not the event metadata described here — and is the standard way for a
tool to return files or UI data to the frontend. See [Rich Content](../rich-content.md).

## Compass turn behavior

Compass loads one versioned metadata snapshot and creates a
`SessionMetadataWorkspace`. Rule discovery, glossary recall, budget accounting,
hooks, and same-process SDK helpers share its dynamic read-only view. Internal
mutations append typed operations to an in-memory journal and perform no
database I/O.

An explicit SDK metadata write from a hook is still persisted immediately. The
workspace rebases the returned session and replays its staged internal
operations, so both changes remain visible. At normal turn completion, Compass
applies the journal to the latest stored revision and makes one conditional
metadata update. A clean turn makes no internal metadata update. On a revision
conflict, Compass re-reads the session, replays its staged operations on the
fresh snapshot, and retries the commit, a bounded number of times.

```text
API event     Compass          Workspace              Session store
    |            |                 |                         |
    |----------->| load session    |                         |
    |            |---------------->| base snapshot + revision|
    |            | match / recall  |                         |
    |            |---------------->| stage typed operations  |
    |            | SDK user set ---------------------------->|
    |            |                 |<---- rebase result -----|
    |            | respond         |                         |
    |<--- ready -|                 |                         |
    |            | commit journal  |---- CAS revision ------>|
    |            |                 |<--- committed session ---|
```

The terminal `ready` event means response streaming has ended. Post-response
compaction, pruning, usage collection, and the metadata commit can still be
finishing. The internal durability boundary is completion of
`CompassEngine.process()`. A successor turn in the same server process waits
for that finalization; an unrelated external read made immediately after
observing `ready` may briefly see the preceding internal snapshot.

## Failure and recovery

When a turn is cancelled or fails, Compass discards the adaptive state it
staged during that turn. Usage that was already incurred may still be kept
through one accounting-only commit. Public SDK writes and emitted events have
already taken effect externally, so they are not rolled back.

The Compass input checkpoint identifies the most recent customer message whose
adaptive state was committed. It includes the event ID, offset, and a hash of
the consumed text so an in-place voice transcript amendment is detected even
when its ID and offset do not change. Event deletion marks the session
`needs_rebuild`; a missing checkpoint event is also detected on the next load.
Compass then clears its bounded adaptive working sets and reconstructs them
from the surviving compacted interaction and subsequent customer input.

## Event update safety

Internal event annotations are included in the event's initial creation
whenever possible. Later voice confirmation and delivery annotations use a
metadata-revision compare-and-swap patch. Public event set/unset operations use
the same revision mechanism, so an application annotation, speculative flag,
and spoken-text update cannot silently replace one another.

Tool-result metadata follows the same ownership rule. Tool-authored result
metadata stays at the top level, while a Parlant-generated tool failure is
stored at `result.metadata.__parlant__.error_details`.
