# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn-scoped, conflict-safe staging for Compass session metadata.

Synopsis
========
``SessionMetadataWorkspace`` gives one Compass ``initialize()`` or ``process()``
call a shared, read-your-writes metadata view. It is the required
``EngineContext.session_metadata`` value: Compass components stage typed
operations in memory and the engine commits their journal once at the call
boundary. There is no direct-persistence fallback for Compass internal metadata.

Motivation
==========
Rule matching, glossary recall, SDK hooks, and usage accounting can run in
overlapping asyncio tasks. Previously, each component copied the session's
complete metadata mapping and persisted its copy. A later write could therefore
erase an unrelated earlier write. Moving every internal value below one
``__parlant__`` root would make that failure larger unless writes also stopped
replacing stale snapshots.

How it works
============
The workspace owns a base metadata snapshot and its metadata-specific revision,
an ordered operation journal, and a derived current view. Staging is
synchronous and performs no store I/O. At commit, the journal is replayed on the
base and conditionally written at the base revision. Each revision conflict
causes a fresh read followed by semantic replay on the newer snapshot, up to the
configured attempt limit. A workspace whose journal is empty closes successfully
without a metadata write.

Terminology
===========
* **Base snapshot**: metadata and revision last known to be durable.
* **View**: immutable-by-contract mapping derived from base plus the journal.
* **Operation journal**: ordered typed intent, never a stale final snapshot.
* **Rebase**: replace the base with a newer version and replay local intent.
* **Behavioral operation**: adaptive Compass state discarded on failure.
* **Accounting operation**: incurred usage/budget state retained on failure.
* **Infrastructure operation**: a reserved outcome category retained by
  behavioral rollback. No current producer stages this category; Compass
  checkpoints and rebuild markers are behavioral operations.
* **Commit**: zero writes for an empty journal; otherwise one successful
  revision-checked update, potentially preceded by conflicting attempts.
* **Behavioral rollback**: discard behavioral intent while leaving the
  workspace active so retained accounting can still commit.
* **Full rollback**: close the workspace without committing its remaining
  journal.

Successful turn
===============
::

    Compass          Workspace             SessionStore
       | read Session(rev=4) |                  |
       |--------------------------------------->|
       |<------------- Session(rev=4) ----------|
       | create workspace    |                  |
       |-------------------->|                  |
       | stage rules         |                  |
       |-------------------->|                  |
       | stage glossary      |                  |
       |-------------------->|                  |
       | SDK reads view      |                  |
       |<--------------------|                  |
       | commit              |                  |
       |-------------------->| CAS rev 4 -> 5   |
       |                     |----------------->|
       |                     |<-----------------|
       |<-- committed Session(rev=5) -----------|

Revision conflict
=================
::

    Worker A          Store          Worker B workspace
       | CAS 7 -> 8     |                    |
       |--------------->|                    |
       |                |<----- CAS at 7 -----|
       |                |------ conflict ---->|
       |                |<----- read latest --|
       |                |------ rev 8 -------->|
       |                |   replay journal     |
       |                |<----- CAS 8 -> 9 ----|
       |                |------ success ------>|

SDK refresh and write
=====================
::

    SDK accessor          Workspace             Store
         | get(key)           |                   |
         |-------------------------------------->|
         |<---------- versioned Session ---------|
         | rebase             |                   |
         |------------------->|                   |
         |<-- merged value ---|                   |
         | set(user_key)      |                   |
         |-------------------------------------->|
         |<---------- versioned Session ---------|
         | rebase             |                   |
         |------------------->|                   |

Cancellation and accounting-only finalization
=============================================
::

    Engine                Workspace                Store
       | stage behavior      |                       |
       |-------------------->|                       |
       | exception/cancel    |                       |
       | error hooks (exception only; workspace active)
       | cancel + await tracked SDK tasks            |
       | drop behavioral     |                       |
       |-------------------->|                       |
       | collect usage       |                       |
       | stage accounting    |                       |
       |-------------------->|                       |
       | commit retained     |                       |
       |-------------------->|                       |
       |                     | close if clean (no I/O)|
       |                     | otherwise bounded ---->|
       |                     |   CAS replay attempts  |
       |<---------------- committed -----------------|

Detached task after finalization
================================
::

    Engine              Workspace          Untracked escaped task
       | cancel + await tracked SDK tasks             |
       | commit/rollback   |                         |
       |------------------>| CLOSED                  |
       |                   |<-- stage operation -----|
       |                   |--- lifecycle error ---->|
       |                   |<-- synchronous read ----|
       |                   |--- final snapshot ----->|

Flow and edge cases
===================
The engine creates one workspace per Compass initialization or process call,
passes it as the required ``EngineContext.session_metadata`` field, and installs
its view on ``EngineContext.session``. Compass has no optional workspace path
and no direct-store fallback. ``EntityContext`` uses a context variable, so
child asyncio tasks and same-process integrated tools inherit the same workspace
reference. Synchronous staging is atomic under the current single-event-loop
execution model because it contains no await. Calling staging methods from
worker threads is unsupported; doing so in the future requires an explicit
lock.

Rebases are revision-monotonic, so two asynchronous SDK refreshes completing
out of order cannot move the base backward. Every operation defensively copies
nested values. Double commit, mutation after finalization, malformed internal
metadata, and exhausted compare-and-swap retries raise explicit errors.

Behavioral rollback filters the journal and reconstructs the view from the
latest base plus retained accounting intent. It cannot undo public SDK writes,
tool side effects, emitted events, or any other already-durable external
effect. The engine keeps the workspace active through error hooks. Before
abandoned-turn finalization it cancels and awaits SDK retriever tasks registered
in ``ResponseState.background_tasks``; only unregistered tasks can escape that
cleanup. It then discards behavioral operations, stages final incurred usage,
and commits retained accounting before closing the workspace.

The terminal ``ready`` event currently precedes Compass post-response
finalization. Therefore the workspace's durability boundary is completion of
``CompassEngine.process()``, not observation of ``ready``. Same-process session
restart waits for the prior task to unwind, preventing a successor turn from
overtaking the commit.

The fresh-data contract deliberately has no legacy dual reads, data migration,
or direct-persistence compatibility path. All Compass producers and consumers
use the namespaced workspace schema together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import cast

from parlant.core.common import JSONSerializable
from parlant.core import metadata as metadata_utils
from parlant.core.sessions import (
    MetadataRevisionConflict,
    Session,
    SessionStore,
)


_DISCOVERED_RULE_IDS_PATH = ("compass", "discovered_rule_ids")
_RULES_EVER_MATCHED_IN_SESSION_PATH = ("compass", "rules_ever_matched_in_session")
_EVICTED_DISCOVERED_RULES_PATH = ("compass", "evicted_discovered_rules")
_LAST_RULE_PRUNING_OFFSET_PATH = ("compass", "last_discovered_rule_pruning_offset")
_SESSION_TERMS_PATH = ("compass", "session_terms")
_USAGE_PATH = ("usage",)


class WorkspaceLifecycleError(RuntimeError):
    """Raised when a caller violates the workspace lifecycle."""


class WorkspaceState(Enum):
    """Lifecycle states of one turn-scoped workspace."""

    ACTIVE = auto()
    """The workspace accepts staging and rebase operations."""

    COMMITTED = auto()
    """The selected journal was durably committed, or was clean."""

    ROLLED_BACK = auto()
    """The workspace was closed without committing its remaining journal."""


class MetadataOperationCategory(Enum):
    """Outcome category used to retain or discard staged operations."""

    BEHAVIORAL = auto()
    """Adaptive Compass state that failed/cancelled turns discard."""

    ACCOUNTING = auto()
    """Usage or budget state representing already-incurred work."""

    INFRASTRUCTURE = auto()
    """Reserved retained category; no current producer stages this category."""


class WorkspaceMetadataView(Mapping[str, JSONSerializable]):
    """Read-only dynamic mapping over a workspace's current metadata.

    Nested values are returned as defensive copies. Retaining this object is
    safe: while active it follows the workspace's read-your-writes state, and
    after finalization it exposes the final immutable snapshot.
    """

    def __init__(self, workspace: SessionMetadataWorkspace) -> None:
        """Bind the view to one workspace.

        Args:
            workspace: Workspace whose current mapping is exposed.

        Side effects:
            None. This method performs no I/O.
        """

        self._workspace = workspace

    def __getitem__(self, key: str) -> JSONSerializable:
        """Return a defensive copy of one top-level value.

        Args:
            key: Top-level metadata key.

        Returns:
            A detached JSON-serializable value.

        Raises:
            KeyError: If the key is absent.
        """

        return deepcopy(self._workspace._current[key])

    def __iter__(self) -> Iterator[str]:
        """Iterate over a stable snapshot of current top-level keys.

        Returns:
            Iterator detached from later workspace mutations.
        """

        return iter(tuple(self._workspace._current))

    def __len__(self) -> int:
        """Return the current number of top-level metadata keys."""

        return len(self._workspace._current)


@dataclass(frozen=True)
class MetadataOperation(ABC):
    """One replayable metadata mutation in the ordered workspace journal."""

    category: MetadataOperationCategory
    """Outcome category controlling rollback retention."""

    @abstractmethod
    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Apply this operation to a metadata snapshot.

        Args:
            metadata: Latest complete metadata mapping.

        Returns:
            A new complete metadata mapping.

        Side effects:
            None. Implementations are synchronous, deterministic, and perform
            no I/O.
        """


@dataclass(frozen=True)
class SetInternalMetadataOperation(MetadataOperation):
    """Set one internal path to a defensively copied value."""

    path: tuple[str, ...]
    value: JSONSerializable

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Set the configured path on ``metadata`` and return a new mapping."""

        return metadata_utils.set_internal_value(metadata, self.path, self.value)


@dataclass(frozen=True)
class UnsetInternalMetadataOperation(MetadataOperation):
    """Remove one internal path if it exists."""

    path: tuple[str, ...]

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Remove the configured path from ``metadata`` and return a new mapping."""

        return metadata_utils.unset_internal_value(metadata, self.path)


@dataclass(frozen=True)
class RuleDiscoveryOperation(MetadataOperation):
    """Union discovered rules and clear their prior eviction entries."""

    rule_ids: frozenset[str]

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Replay rule discovery against the latest rule state."""

        discovered = {
            str(rule_id)
            for rule_id in cast(
                Sequence[str],
                metadata_utils.internal_value(metadata, _DISCOVERED_RULE_IDS_PATH, []),
            )
        }
        discovered.update(self.rule_ids)

        eviction_ledger = dict(
            cast(
                Mapping[str, int],
                metadata_utils.internal_value(metadata, _EVICTED_DISCOVERED_RULES_PATH, {}),
            )
        )
        for rule_id in self.rule_ids:
            eviction_ledger.pop(rule_id, None)

        result = metadata_utils.set_internal_value(
            metadata,
            _DISCOVERED_RULE_IDS_PATH,
            sorted(discovered),
        )
        return metadata_utils.set_internal_value(
            result,
            _EVICTED_DISCOVERED_RULES_PATH,
            cast(JSONSerializable, eviction_ledger),
        )


@dataclass(frozen=True)
class RuleMatchOperation(MetadataOperation):
    """Union rule ids into the session's matched-rule ledger.

    The ledger is a verdict-level latch: a rule that has turn-matched at
    least once this session stays in it for the session's lifetime (eviction
    from the discovered working set deliberately does not erase history).
    Compass satisfies dependency relationships from this ledger."""

    rule_ids: frozenset[str]

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Union this operation's rule ids into the ledger in ``metadata``.

        Args:
            metadata: Latest complete metadata mapping.

        Returns:
            A new mapping whose matched-rule ledger includes ``rule_ids``.
        """
        matched = {
            str(rule_id)
            for rule_id in cast(
                Sequence[str],
                metadata_utils.internal_value(metadata, _RULES_EVER_MATCHED_IN_SESSION_PATH, []),
            )
        }
        matched.update(self.rule_ids)

        return metadata_utils.set_internal_value(
            metadata,
            _RULES_EVER_MATCHED_IN_SESSION_PATH,
            sorted(matched),
        )


@dataclass(frozen=True)
class RulePruningOperation(MetadataOperation):
    """Evict selected rules and monotonically advance pruning state."""

    rule_ids: frozenset[str]
    event_offset: int

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Replay rule pruning without erasing unrelated concurrent discovery."""

        discovered = {
            str(rule_id)
            for rule_id in cast(
                Sequence[str],
                metadata_utils.internal_value(metadata, _DISCOVERED_RULE_IDS_PATH, []),
            )
        }
        discovered.difference_update(self.rule_ids)

        eviction_ledger = dict(
            cast(
                Mapping[str, int],
                metadata_utils.internal_value(metadata, _EVICTED_DISCOVERED_RULES_PATH, {}),
            )
        )
        for rule_id in self.rule_ids:
            eviction_ledger[rule_id] = max(
                self.event_offset,
                int(eviction_ledger.get(rule_id, -1)),
            )

        previous_offset = int(
            cast(
                int,
                metadata_utils.internal_value(metadata, _LAST_RULE_PRUNING_OFFSET_PATH, -1),
            )
        )
        result = metadata_utils.set_internal_value(
            metadata,
            _DISCOVERED_RULE_IDS_PATH,
            sorted(discovered),
        )
        result = metadata_utils.set_internal_value(
            result,
            _EVICTED_DISCOVERED_RULES_PATH,
            cast(JSONSerializable, eviction_ledger),
        )
        return metadata_utils.set_internal_value(
            result,
            _LAST_RULE_PRUNING_OFFSET_PATH,
            max(previous_offset, self.event_offset),
        )


@dataclass(frozen=True)
class GlossaryUpdateOperation(MetadataOperation):
    """Merge glossary hits and conditionally remove stale terms."""

    hits: Mapping[str, int]
    remove_if_not_newer: Mapping[str, int]

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Replay glossary changes while preserving newer concurrent hits."""

        terms = {
            str(term_id): int(offset)
            for term_id, offset in cast(
                Mapping[str, int],
                metadata_utils.internal_value(metadata, _SESSION_TERMS_PATH, {}),
            ).items()
        }

        for term_id, observed_offset in self.remove_if_not_newer.items():
            if terms.get(term_id, observed_offset + 1) <= observed_offset:
                terms.pop(term_id, None)

        for term_id, hit_offset in self.hits.items():
            terms[term_id] = max(hit_offset, terms.get(term_id, -1))

        return metadata_utils.set_internal_value(
            metadata,
            _SESSION_TERMS_PATH,
            cast(JSONSerializable, terms),
        )


def _usage_ledger(
    metadata: Mapping[str, JSONSerializable],
) -> tuple[dict[str, dict[str, JSONSerializable]], dict[str, dict[str, int]]]:
    """Copy the records and baselines from the current internal usage ledger."""

    usage = cast(
        Mapping[str, JSONSerializable], metadata_utils.internal_value(metadata, _USAGE_PATH, {})
    )
    records = {
        str(trace_id): deepcopy(dict(cast(Mapping[str, JSONSerializable], record)))
        for trace_id, record in cast(
            Mapping[str, JSONSerializable],
            usage.get("records", {}),
        ).items()
    }
    baselines = {
        str(key): {
            "sequence": int(
                cast(
                    int,
                    cast(Mapping[str, JSONSerializable], baseline).get("sequence", 0),
                )
            )
        }
        for key, baseline in cast(
            Mapping[str, JSONSerializable],
            usage.get("baselines", {}),
        ).items()
    }
    return records, baselines


def _latest_usage_sequence(records: Mapping[str, Mapping[str, JSONSerializable]]) -> int:
    """Return the greatest durable usage-record sequence."""

    return max(
        (int(cast(int, record.get("sequence", 0))) for record in records.values()),
        default=0,
    )


@dataclass(frozen=True)
class BudgetInitializationOperation(MetadataOperation):
    """Initialize one budget baseline only when it is absent."""

    budget_key: str

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Replay conditional baseline initialization on the latest ledger."""

        records, baselines = _usage_ledger(metadata)
        baselines.setdefault(
            self.budget_key,
            {"sequence": _latest_usage_sequence(records)},
        )
        return metadata_utils.set_internal_value(
            metadata,
            _USAGE_PATH,
            cast(JSONSerializable, {"records": records, "baselines": baselines}),
        )


@dataclass(frozen=True)
class BudgetResetOperation(MetadataOperation):
    """Reset one budget baseline at the latest replay-time sequence."""

    budget_key: str

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Replay a budget reset without replacing unrelated baselines."""

        records, baselines = _usage_ledger(metadata)
        baselines[self.budget_key] = {"sequence": _latest_usage_sequence(records)}
        return metadata_utils.set_internal_value(
            metadata,
            _USAGE_PATH,
            cast(JSONSerializable, {"records": records, "baselines": baselines}),
        )


@dataclass(frozen=True)
class UsageRecordOperation(MetadataOperation):
    """Append one idempotent trace record with a replay-time sequence."""

    trace_id: str
    timestamp_utc: str
    models: Mapping[str, JSONSerializable]
    costs: JSONSerializable
    pricing_error: str | None

    def apply(
        self,
        metadata: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Append the trace unless a prior durable attempt already recorded it."""

        records, baselines = _usage_ledger(metadata)
        if self.trace_id not in records:
            records[self.trace_id] = {
                "sequence": _latest_usage_sequence(records) + 1,
                "timestamp_utc": self.timestamp_utc,
                "models": deepcopy(dict(self.models)),
                "costs": deepcopy(self.costs),
                "pricing_error": self.pricing_error,
            }
        return metadata_utils.set_internal_value(
            metadata,
            _USAGE_PATH,
            cast(JSONSerializable, {"records": records, "baselines": baselines}),
        )


class SessionMetadataWorkspace:
    """Own one Compass call's metadata view, journal, and CAS lifecycle.

    The engine creates this class with the loaded ``Session`` and authoritative
    ``SessionStore``. Components stage operations synchronously. The engine
    alone calls :meth:`commit` or :meth:`rollback`.
    """

    def __init__(
        self,
        session: Session,
        store: SessionStore,
        *,
        max_commit_attempts: int = 4,
    ) -> None:
        """Create an active workspace from a versioned session snapshot.

        Args:
            session: Session loaded from the authoritative writable store.
            store: Store providing revision-checked metadata commits.
            max_commit_attempts: Positive bound including the first CAS attempt.

        Raises:
            ValueError: If ``max_commit_attempts`` is not positive.

        Side effects:
            Copies metadata into memory. No store I/O occurs.
        """

        if max_commit_attempts < 1:
            raise ValueError("max_commit_attempts must be positive")

        self._store = store
        self._session = session
        self._base_metadata = deepcopy(dict(session.metadata))
        self._base_revision = session.metadata_revision
        self._operations: list[MetadataOperation] = []
        self._current = deepcopy(self._base_metadata)
        self._state = WorkspaceState.ACTIVE
        self._max_commit_attempts = max_commit_attempts
        self._view = WorkspaceMetadataView(self)

    @property
    def state(self) -> WorkspaceState:
        """Return the current lifecycle state without mutating it."""

        return self._state

    @property
    def view(self) -> WorkspaceMetadataView:
        """Return the stable dynamic read-only metadata view."""

        return self._view

    @property
    def session(self) -> Session:
        """Return current first-class session fields with the workspace view."""

        return replace(
            self._session,
            metadata=self._view,
            metadata_revision=self._base_revision,
        )

    @property
    def is_dirty(self) -> bool:
        """Return whether at least one staged operation remains."""

        return bool(self._operations)

    def get_internal(
        self,
        path: metadata_utils.MetadataPath,
        default: JSONSerializable = None,
    ) -> JSONSerializable:
        """Read one internal path from the current staged view.

        Args:
            path: Path relative to ``__parlant__``.
            default: Detached value returned when absent.

        Returns:
            A defensive copy of the current value.

        Side effects:
            None. Reads are allowed after finalization.
        """

        return metadata_utils.internal_value(self._current, path, default)

    def set_internal(
        self,
        path: metadata_utils.MetadataPath,
        value: JSONSerializable,
        *,
        category: MetadataOperationCategory = MetadataOperationCategory.BEHAVIORAL,
    ) -> None:
        """Stage a generic internal path assignment.

        Args:
            path: Non-empty path relative to ``__parlant__``.
            value: JSON-serializable value copied into the operation.
            category: Outcome category used by rollback.

        Raises:
            WorkspaceLifecycleError: If the workspace is not active.
            ValueError: If the path is empty or malformed.

        Side effects:
            Appends one operation and synchronously rebuilds the current view.
            Performs no I/O.
        """

        self._append(
            SetInternalMetadataOperation(
                category=category,
                path=tuple(path),
                value=deepcopy(value),
            )
        )

    def unset_internal(
        self,
        path: metadata_utils.MetadataPath,
        *,
        category: MetadataOperationCategory = MetadataOperationCategory.BEHAVIORAL,
    ) -> None:
        """Stage removal of one internal path.

        Args:
            path: Non-empty path relative to ``__parlant__``.
            category: Outcome category used by rollback.

        Raises:
            WorkspaceLifecycleError: If the workspace is not active.
            ValueError: If the path is empty or malformed.

        Side effects:
            Appends one operation and updates the view without I/O.
        """

        self._append(
            UnsetInternalMetadataOperation(
                category=category,
                path=tuple(path),
            )
        )

    def stage_rule_discovery(self, rule_ids: Sequence[str]) -> None:
        """Stage idempotent discovery of Compass rule IDs.

        Args:
            rule_ids: Rule IDs admitted to the session working set.

        Raises:
            WorkspaceLifecycleError: If ``rule_ids`` is non-empty and the
                workspace is not active.

        Side effects:
            Unions IDs and removes their eviction entries in the current view.
            Performs no I/O. An empty sequence is a no-op and does not inspect
            lifecycle state.
        """

        if rule_ids:
            self._append(
                RuleDiscoveryOperation(
                    category=MetadataOperationCategory.BEHAVIORAL,
                    rule_ids=frozenset(rule_ids),
                )
            )

    def stage_rule_matches(self, rule_ids: Sequence[str]) -> None:
        """Stage idempotent recording of turn-matched Compass rule IDs.

        Args:
            rule_ids: Rule IDs that turn-matched (verdict-level, recorded
                before priority resolution).

        Raises:
            WorkspaceLifecycleError: If ``rule_ids`` is non-empty and the
                workspace is not active.

        Side effects:
            Unions IDs into the matched-rule ledger in the current view.
            Performs no I/O. An empty sequence is a no-op and does not inspect
            lifecycle state.
        """

        if rule_ids:
            self._append(
                RuleMatchOperation(
                    category=MetadataOperationCategory.BEHAVIORAL,
                    rule_ids=frozenset(rule_ids),
                )
            )

    def stage_rule_pruning(self, rule_ids: Sequence[str], event_offset: int) -> None:
        """Stage coupled Compass rule eviction bookkeeping.

        Args:
            rule_ids: IDs selected for eviction.
            event_offset: Monotonic session event offset of the decision.

        Raises:
            WorkspaceLifecycleError: If ``rule_ids`` is non-empty and the
                workspace is not active.

        Side effects:
            Updates discovered IDs, eviction ledger, and pruning offset
            synchronously. Performs no I/O. An empty sequence is a no-op and
            does not inspect lifecycle state.
        """

        if rule_ids:
            self._append(
                RulePruningOperation(
                    category=MetadataOperationCategory.BEHAVIORAL,
                    rule_ids=frozenset(rule_ids),
                    event_offset=event_offset,
                )
            )

    def stage_glossary_update(
        self,
        *,
        hits: Mapping[str, int] = {},
        remove_if_not_newer: Mapping[str, int] = {},
    ) -> None:
        """Stage glossary hits and safe conditional removals.

        Args:
            hits: Term IDs mapped to their newest hit offsets.
            remove_if_not_newer: Term IDs mapped to offsets observed when
                removal was selected. A newer concurrent hit survives replay.

        Raises:
            WorkspaceLifecycleError: If either input is non-empty and the
                workspace is not active.

        Side effects:
            Appends one typed operation and updates the view without I/O. Empty
            inputs are a no-op and do not inspect lifecycle state.
        """

        if hits or remove_if_not_newer:
            self._append(
                GlossaryUpdateOperation(
                    category=MetadataOperationCategory.BEHAVIORAL,
                    hits=dict(hits),
                    remove_if_not_newer=dict(remove_if_not_newer),
                )
            )

    def stage_budget_initialization(self, budget_key: str) -> None:
        """Stage initialize-if-absent semantics for one budget baseline.

        Args:
            budget_key: Stable session-budget identifier.

        Raises:
            WorkspaceLifecycleError: If the workspace is finalized.

        Side effects:
            Adds an accounting operation without performing I/O.
        """

        self._append(
            BudgetInitializationOperation(
                category=MetadataOperationCategory.ACCOUNTING,
                budget_key=budget_key,
            )
        )

    def stage_budget_reset(self, budget_key: str) -> None:
        """Stage a reset of one budget baseline.

        Args:
            budget_key: Stable session-budget identifier.

        Raises:
            WorkspaceLifecycleError: If the workspace is finalized.

        Side effects:
            Adds an accounting operation without performing I/O.
        """

        self._append(
            BudgetResetOperation(
                category=MetadataOperationCategory.ACCOUNTING,
                budget_key=budget_key,
            )
        )

    def stage_usage_record(
        self,
        *,
        trace_id: str,
        timestamp_utc: str,
        models: Mapping[str, JSONSerializable],
        costs: JSONSerializable,
        pricing_error: str | None,
    ) -> None:
        """Stage one idempotent usage record for durable turn accounting.

        Args:
            trace_id: Trace identifier used as the idempotency key.
            timestamp_utc: ISO timestamp captured when accounting finalized.
            models: Serialized usage counters keyed by model.
            costs: Serialized server-wide cost snapshot, or ``None`` when unpriced.
            pricing_error: Internal configuration failure for an unpriceable trace.

        Raises:
            WorkspaceLifecycleError: If the workspace is finalized.

        Side effects:
            Adds an accounting operation without performing I/O.
        """

        self._append(
            UsageRecordOperation(
                category=MetadataOperationCategory.ACCOUNTING,
                trace_id=trace_id,
                timestamp_utc=timestamp_utc,
                models=deepcopy(dict(models)),
                costs=deepcopy(costs),
                pricing_error=pricing_error,
            )
        )

    def rebase(self, session: Session) -> bool:
        """Adopt a newer versioned session and replay the local journal.

        Args:
            session: Fresh session returned by a versioned SDK/store operation.

        Returns:
            ``True`` when the base advanced or first-class fields refreshed;
            ``False`` when the snapshot was older and therefore ignored.

        Raises:
            WorkspaceLifecycleError: If the workspace is finalized.

        Side effects:
            Replaces the in-memory base and reconstructs the view. Performs no
            I/O itself.
        """

        self._ensure_active()
        if session.metadata_revision < self._base_revision:
            return False

        self._session = session
        self._base_revision = session.metadata_revision
        self._base_metadata = deepcopy(dict(session.metadata))
        self._rebuild_current()
        return True

    def rollback_behavioral_state(self) -> None:
        """Discard behavioral operations while retaining other categories.

        Raises:
            WorkspaceLifecycleError: If the workspace is finalized.

        Side effects:
            Filters the journal and reconstructs the current view without I/O.
            The workspace remains active so retained accounting can be
            committed.
        """

        self._ensure_active()
        self._operations = [
            operation
            for operation in self._operations
            if operation.category is not MetadataOperationCategory.BEHAVIORAL
        ]
        self._rebuild_current()

    def rollback(self) -> Session:
        """Close the workspace without committing its remaining operations.

        Returns:
            Session carrying the latest rebased durable metadata snapshot.

        Raises:
            WorkspaceLifecycleError: If already finalized.

        Side effects:
            Clears the journal, restores the base view, and marks the workspace
            ``ROLLED_BACK``. Performs no I/O.
        """

        self._ensure_active()
        self._operations.clear()
        self._current = deepcopy(self._base_metadata)
        self._state = WorkspaceState.ROLLED_BACK
        return self.session

    async def commit(self) -> Session:
        """Commit the complete remaining journal with bounded CAS replay.

        Returns:
            Session returned by the successful store commit, carrying the
            workspace view. A clean commit returns without store I/O.

        Raises:
            WorkspaceLifecycleError: If already finalized.
            MetadataRevisionConflict: If every bounded attempt conflicts.
            NotImplementedError: If the configured store lacks atomic commit
                support.
            Exception: Any non-conflict store failure is propagated.

        Idempotency:
            Calling this method twice is rejected. Individual typed operations
            are designed to replay safely after a conflict.

        Side effects:
            Performs at most one store update per attempt and a fresh read only
            after a real conflict. On success, closes the workspace as
            ``COMMITTED``.
        """

        self._ensure_active()
        if not self._operations:
            self._state = WorkspaceState.COMMITTED
            return self.session

        last_conflict: MetadataRevisionConflict | None = None
        for attempt in range(self._max_commit_attempts):
            candidate = self._apply_operations(self._base_metadata)
            try:
                committed = await self._store.commit_session_metadata(
                    self._session.id,
                    self._base_revision,
                    candidate,
                )
            except MetadataRevisionConflict as conflict:
                last_conflict = conflict
                if attempt + 1 == self._max_commit_attempts:
                    break
                latest = await self._store.read_session(self._session.id)
                self.rebase(latest)
                continue

            self._session = committed
            self._base_revision = committed.metadata_revision
            self._base_metadata = deepcopy(dict(committed.metadata))
            self._operations.clear()
            self._current = deepcopy(self._base_metadata)
            self._state = WorkspaceState.COMMITTED
            return self.session

        assert last_conflict is not None
        raise last_conflict

    def _append(self, operation: MetadataOperation) -> None:
        """Append and apply one operation after validating active lifecycle.

        Args:
            operation: Fully copied, deterministic operation.

        Raises:
            WorkspaceLifecycleError: If finalized.
            ValueError: If the operation encounters malformed metadata.

        Side effects:
            Adds the operation only after it applies successfully. Performs no
            I/O.
        """

        self._ensure_active()
        updated = operation.apply(self._current)
        self._operations.append(operation)
        self._current = updated

    def _apply_operations(
        self,
        base: Mapping[str, JSONSerializable],
    ) -> dict[str, JSONSerializable]:
        """Replay the ordered journal on a detached base snapshot.

        Args:
            base: Complete metadata mapping to replay against.

        Returns:
            New complete metadata mapping.

        Side effects:
            None. Performs no I/O and preserves journal order.
        """

        current = deepcopy(dict(base))
        for operation in self._operations:
            current = operation.apply(current)
        return current

    def _rebuild_current(self) -> None:
        """Recompute the view from the current base and ordered journal.

        Side effects:
            Replaces the in-memory derived view. Performs no I/O.
        """

        self._current = self._apply_operations(self._base_metadata)

    def _ensure_active(self) -> None:
        """Require the workspace to be active before a mutating operation.

        Raises:
            WorkspaceLifecycleError: If committed or rolled back.

        Side effects:
            None.
        """

        if self._state is not WorkspaceState.ACTIVE:
            raise WorkspaceLifecycleError(
                f"Session metadata workspace is already {self._state.name.lower()}"
            )
