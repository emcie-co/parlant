# Diagnostics and telemetry

Parlant provides three complementary forms of operational telemetry:

- diagnostic reports for recent engine turns and errors;
- the local `parlant.log` server log;
- optional OpenTelemetry traces, logs, and metrics.

Diagnostic reports are one feature with two outcomes. Every retained engine
turn can be exported in one archive. A report whose outcome is `error` contains
the same trace evidence as a successful report, plus the exception and
traceback. Both outcomes use the same format and API.

## Filesystem configuration

`PARLANT_HOME` is the base directory for persistent Parlant state. If it is not
set, Parlant uses `./parlant-data`, relative to the server process's working
directory. Telemetry artifacts use these defaults:

| Artifact | Directory override | Default |
| --- | --- | --- |
| Error diagnostic reports | `PARLANT_ERROR_REPORTS_DIR` | `${PARLANT_HOME}/diagnostic-reports/errors` |
| Temporary trace export files | `PARLANT_TRACE_REPORTS_DIR` | `${PARLANT_HOME}/diagnostic-reports/traces` |
| Local logs | `PARLANT_LOGS_DIR` | `${PARLANT_HOME}/logs` |

The active local log is `${PARLANT_LOGS_DIR}/parlant.log`.

An artifact-specific variable takes precedence over `PARLANT_HOME` for that
artifact only. This makes it possible to place reports and logs on volumes with
different size, retention, security, or lifecycle policies:

```bash
export PARLANT_HOME=/var/lib/parlant
export PARLANT_ERROR_REPORTS_DIR=/mnt/restricted/parlant/errors
export PARLANT_TRACE_REPORTS_DIR=/mnt/diagnostics/parlant/traces
export PARLANT_LOGS_DIR=/var/log/parlant
```

Relative paths are resolved against the server process's working directory.
Set these variables before starting or importing the server because runtime
paths and the local logger are configured during module initialization.

Error-report and temporary export directories are created with owner-only
permissions (`0700`), and archives with owner read/write permissions (`0600`)
where the platform permits. Local log directories and files use the same
owner-only policy.

## Diagnostic reports

Diagnostic reporting is enabled by default. With the Python SDK:

```python
import parlant.sdk as p

async with p.Server(diagnostic_reports=True) as server:
    ...
```

To disable all diagnostic capture and persistence:

```python
async with p.Server(diagnostic_reports=False) as server:
    ...
```

Only engine turns enter diagnostic history. Ordinary health, listing, and
other HTTP traces are discarded when their final in-memory owner exits.

### Storage and retention

Completed engine turns are kept as bounded snapshots in process memory. Parlant
does not compress or write a successful trace when the turn completes. It
creates the ZIP only if an administrator requests that trace through the export
API, stages it in `PARLANT_TRACE_REPORTS_DIR` while streaming, and deletes it
after the response. ZIP construction runs in the server's worker-thread pool so
compression does not block the async event loop. The in-memory history is lost
when the process exits.

Errors are handled differently: their reports are written to disk right away
and kept there. An error archive is written to `PARLANT_ERROR_REPORTS_DIR` as
soon as the exception is captured, then finalized in place when the turn
completes. This makes the URI in an error event usable
immediately and lets error diagnostics survive a restart.

Retention is the union of two policies:

```text
available diagnostics = newest N in-memory engine turns ∪ newest M error archives
```

- `N` is `PARLANT_TRACE_HISTORY_SIZE`, default `32`.
- `M` is `PARLANT_ERROR_HISTORY_SIZE`, default `8`.

An error counts as one of the newest N turns and has one persistent archive in
the error directory; it is never duplicated in the trace directory. An older
error archive remains available if it is still among the newest M error
outcomes. A successful trace becomes unavailable once its in-memory snapshot
leaves the newest N turns.

Retention is deployment configuration and is controlled only through
environment variables:

```text
PARLANT_TRACE_HISTORY_SIZE=64
PARLANT_ERROR_HISTORY_SIZE=16
```

Set the trace history to `0` to retain only persistent errors. Set the error
history to `0` to remove its additional protection; an error archive remains
on disk only while that error is in the in-memory trace window. Setting both
values to `0` disables diagnostic capture and persistence. Both values must be
non-negative integers.

Error retention is applied after finalization, when a protected error leaves
the trace window, and once at startup. Because the trace window starts empty
after a restart, startup retains the newest M error archives. Files without the
expected report filename or a valid error-report manifest are ignored.

### Export API

The administrator-only debug API exposes both outcomes:

```http
GET /debug/reports
GET /debug/reports?limit=10
GET /debug/reports?outcome=success
GET /debug/reports?outcome=error
GET /debug/reports/{trace_id}
```

The listing is newest first and includes the trace ID, session ID, outcome,
creation/finalization timestamps, archive size, and download URI. The archive
size is `null` for an in-memory successful trace because no ZIP exists yet.
`limit` must be at least `1`. `outcome` accepts `success` or `error`.

`/healthz` also surfaces the most recent reports as a `diagnostic_reports`
list (trace ID, outcome, creation time, and the `/debug/reports/{trace_id}`
download URI), so an operator looking at health can jump straight to the
latest turns. The count is set by `PARLANT_HEALTHZ_REPORT_LINKS` (default 3;
`0` removes the section). The links appear only for callers holding the
`read_health_report` permission — see the note below — and downloading a
report still requires the administrator-only debug API.

### Health endpoint visibility

`/healthz` always answers `200` with a top-level `status`, so load balancers
and uptime probes work unauthenticated. Everything beyond that — per-component
checks, latency percentiles, error classes, and the report links — is shown
only to callers granted the `read_health_report` permission. Under the
production composite policy that means admin callers; the development policy
shows everything.

An initial error archive is downloadable from its event URI immediately, but it
does not appear in the listing until its manifest is finalized. If finalization
fails, that incomplete archive remains directly downloadable through the URI.

The download endpoint returns `application/zip` with a filename of
`{trace_id}.zip`. Downloading a successful trace performs ZIP compression on
demand; concurrent requests use separate temporary files, and each file is
deleted after its response. Invalid, missing, expired, symlinked, and
non-regular candidates return `404` without exposing filesystem paths. Under an
authenticated authorization policy, non-administrators receive `403`.

Example:

```bash
curl -H "Authorization: Bearer $PARLANT_ADMIN_TOKEN" \
  "$PARLANT_URL/debug/reports?outcome=error"

curl -fL -H "Authorization: Bearer $PARLANT_ADMIN_TOKEN" \
  -o report.zip \
  "$PARLANT_URL/debug/reports/6658572371514ad7897b3bef729a4603"
```

### Error event metadata

When an engine error is captured, the emitted status event contains an
immediately usable URI:

```json
{
  "metadata": {
    "__parlant__": {
      "schema_version": 1,
      "diagnostic_report_uri": "/debug/reports/6658572371514ad7897b3bef729a4603"
    }
  }
}
```

The URI is omitted if diagnostic reporting is disabled or the initial archive
could not be written. Successful events do not need embedded report metadata;
administrators discover successful reports through the listing endpoint.

### Archive contents

Every finalized ZIP contains:

- `manifest.json`: schema version, trace/session IDs, Parlant version,
  timestamps, outcome, completeness, exception summary, and member list;
- `trace.json`: the local exporter-independent span timeline;
- `logs.jsonl`: application logger calls routed through `container[Logger]`
  while the trace was live;
- `session.json` and `events.jsonl` when persisted session state was available.

Error reports additionally contain `traceback.txt` with the full formatted
traceback. Successful reports do not synthesize an exception or traceback.

Diagnostic log capture is intentionally independent of the server output log
level. TRACE and DEBUG calls remain in `logs.jsonl` even when `parlant.log` or
an OpenTelemetry sink is configured for INFO or higher. Messages produced by
the diagnostic-report service itself are excluded to prevent feedback loops.

At the engine exception boundary, Parlant first writes an incomplete error
archive (`manifest.complete` is `false`) before emitting its URI. When the last
trace owner exits, the same archive is atomically replaced with a complete
version containing closing spans, late logs, and session events. A successful
trace is stored only as a detached completed in-memory snapshot; its complete
archive is assembled on demand from that snapshot.

Diagnostic capture errors never replace the original turn result. Capture,
finalization, and retention errors are sent to the normal server logger.

## Health endpoint

`GET /healthz` reports one top-level `status` and a `checks` object with one
entry per component. The components that can change the top-level status are:

- `engine`: the success rate and p95 duration of recent turns.
- `nlp`: the success rate and p50/p95 latency of model requests, per model
  route, with a per-agent breakdown inside each route.
- `event_loop`: how much of the last five minutes (a rolling window) the
  server's event loop spent lagging, and the peak lag in that time. Lag means
  something is blocking the loop, usually a tool doing synchronous work. The
  status follows the time spent lagging, so one brief stall shows as the peak
  (`latency_ms`) without changing the status; `lag_share` is the share of the
  window spent in lags at the degraded cutoff or above.
- `tunnel`: whether the connection to the Parlant Dashboard is up, for
  managed projects only.

Each component is `healthy`, `degraded` or `unhealthy`. The top-level status
is the worst of them. The `engine` and `nlp` entries also list `reasons`: each
threshold that was crossed, with the observed value and the threshold it was
compared against, so an operator can see why the status changed without
knowing the defaults.

Who sees the `checks` detail is covered under
[Health endpoint visibility](#health-endpoint-visibility).

### Status thresholds

The cutoffs live in one `HealthThresholds` value bound on the container. Bind
your own in `configure_container` to change them. Anything you leave out keeps
its default.

| Check | Threshold | Degraded | Unhealthy |
| --- | --- | --- | --- |
| `engine` | success rate below | 0.95 | 0.80 |
| `engine` | turn p95 latency at or above | 30 s | 60 s |
| `nlp` | success rate below | 0.95 | 0.80 |
| `nlp` | request p50 latency at or above | 20 s | 30 s |
| `nlp` | request p95 latency at or above | 45 s | 60 s |
| `event_loop` | lags at or above this add up to 1% of the window (3 s) | 200 ms | 500 ms |

The `nlp` defaults above apply to every model route. Four routes on the
response path have tighter per-route overrides:

| Route | p50 degraded / unhealthy | p95 degraded / unhealthy |
| --- | --- | --- |
| `matching.rank` | 8 s / 12 s | 15 s / 20 s |
| `matching.distill` | 8 s / 12 s | 15 s / 20 s |
| `review.message` | 10 s / 15 s | 20 s / 25 s |
| `review.tools` | 10 s / 15 s | 20 s / 25 s |

Say your agents use a slow reasoning model for message generation, so a 60
second p95 is normal, while you want the health check to react quickly to a
backlog on the event loop. Bind thresholds that say so:

```python
import parlant.sdk as p


async def configure_container(container: p.Container) -> p.Container:
    container[p.HealthThresholds] = p.HealthThresholds(
        nlp=p.NLPHealthThresholds(
            degraded_p95_ms=90_000.0,
            unhealthy_p95_ms=120_000.0,
            routes={
                # Keep the default matching cutoffs; only review gets more room.
                **p.default_nlp_route_thresholds(),
                "review.message": p.RouteThresholds(
                    degraded_p95_ms=40_000.0,
                    unhealthy_p95_ms=60_000.0,
                ),
            },
        ),
        event_loop=p.EventLoopHealthThresholds(
            degraded_lag_ms=100.0,
            unhealthy_lag_ms=250.0,
        ),
    )
    return container


async def main() -> None:
    async with p.Server(configure_container=configure_container) as server:
        ...
```

A few rules to keep in mind:

- Passing `routes` replaces the whole override table. Spread
  `p.default_nlp_route_thresholds()` first, as above, to keep the defaults
  for the routes you don't mention.
- A field left `None` in a `RouteThresholds` override falls back to the
  view-wide `nlp` value for that route.
- Setting a p50 cutoff to `None` turns that comparison off; p95 and success
  rate cutoffs are always applied.
- `p.EventLoopHealthThresholds(lag_share=...)` sets how much of the window
  the lag must add up to, from 0 to 1 (default `0.01`). Each late measurement
  counts for as long as it was late, so one long stall weighs as much as many
  short ones. `lag_share=0.0` makes any single lag at a cutoff count.
- The thresholds only change how observations are classified. They do not
  change what is measured or how long observations are kept.

## Local logs and rotation

Parlant writes local server logs to:

```text
${PARLANT_LOGS_DIR}/parlant.log
```

Size-based rotation is enabled by default:

| Setting | Environment variable | Default |
| --- | --- | --- |
| Maximum active-file size | `PARLANT_LOG_ROTATION_MAX_BYTES` | `10485760` (10 MiB) |
| Rotated backups | `PARLANT_LOG_ROTATION_BACKUP_COUNT` | `5` |

When the next log record would push `parlant.log` past the configured size,
the logger rotates before writing it:

```text
parlant.log      current
parlant.log.1    newest backup
parlant.log.2
...
parlant.log.5    oldest backup
```

The oldest backup is removed when a new rollover exceeds the configured backup
count. Set the maximum size to `0` to disable rotation. When rotation is
enabled, the backup count must be at least `1`; all rotation values must be
non-negative.

Example for a small ephemeral volume:

```bash
export PARLANT_LOGS_DIR=/tmp/parlant-logs
export PARLANT_LOG_ROTATION_MAX_BYTES=2097152
export PARLANT_LOG_ROTATION_BACKUP_COUNT=2
```

The rotating file handler is process-safe only within one Parlant process. Do
not point multiple server processes at the same `parlant.log`; give each process
its own log directory or use a centralized collector.

If `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` is configured, application logs are sent
through the OpenTelemetry/WebSocket logging setup instead of the local file
sink. Direct server lifecycle messages still use the local file/console logger.

## OpenTelemetry

Parlant enables OpenTelemetry exporters when their standard endpoint variables
are configured:

- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
- `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`
- `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`

Diagnostic reports do not depend on an external tracing exporter. Their local
span timeline and trace-scoped log buffer remain available with the default
local tracer and also operate alongside OpenTelemetry.

In-memory diagnostic history, report archives, and logs may contain customer
messages, model/tool data, metadata, and exception details. Treat the server
process and telemetry directories as sensitive, restrict API administration
credentials, and align external volume retention and backup policies with your
deployment's data-handling requirements.
