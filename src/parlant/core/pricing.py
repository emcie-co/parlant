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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, cast

from parlant.core.common import JSONSerializable
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace
from parlant.core import metadata as metadata_utils
from parlant.core.nlp.common import UsageInfo, parse_reported_cost
from parlant.core.sessions import SessionId, SessionStore
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter


_MILLION = Decimal(1_000_000)
_REPORTED_COST_CURRENCY = "USD"


def _reported_cost(usage: UsageInfo) -> Decimal | None:
    return parse_reported_cost(usage.extra.get("reported_cost"))


def _is_free(usage: UsageInfo) -> bool:
    """A usage served entirely from cache: nothing generated, no cost reported.

    It cost nothing, so it says nothing about whether its trace can be priced or
    whether its model reports costs."""
    return (
        _reported_cost(usage) is None
        and usage.output_tokens == 0
        and usage.input_tokens == usage.cached_input_tokens
    )


class PricingConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPricing:
    """Per-million prices for one concrete provider model.

    ``extra`` prices additional billable meters reported in
    ``UsageInfo.extra``. Meters the adapters emit today:

    - ``cache_write_5m_input_tokens`` / ``cache_write_1h_input_tokens`` —
      Anthropic prompt-cache writes by TTL (billed at 1.25x / 2x input).
    - ``cache_write_input_tokens`` — OpenAI cache writes (1.25x).
    - ``cache_storage_token_hours`` — Gemini explicit-cache storage, accrued
      once at cache creation (price is the provider's per-token-hour rate).
    - ``reasoning_tokens`` — informational subset of output; price it only if
      the provider bills reasoning separately from output.

    ``reported_cost`` is not a meter: it is an authoritative provider-reported
    cost consumed by :meth:`Pricing.calculate_model` directly.
    """

    input_tokens: Decimal
    cached_input_tokens: Decimal
    output_tokens: Decimal
    extra: Mapping[str, Decimal] = field(default_factory=dict)

    def calculate(self, usage: UsageInfo) -> Decimal:
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        uncached = usage.input_tokens - cached

        total = (
            Decimal(uncached) * self.input_tokens
            + Decimal(cached) * self.cached_input_tokens
            + Decimal(usage.output_tokens) * self.output_tokens
        )

        for key, price in self.extra.items():
            value = usage.extra.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += Decimal(str(value)) * price

        return total / _MILLION


def split_usage_key(key: str) -> tuple[str, str] | None:
    """Split a ``provider:model`` usage key, or None for a key with no provider."""

    provider, separator, model = key.partition(":")
    if not separator or not provider or not model:
        return None
    return provider, model


def costs_per_provider(per_model: Mapping[str, Decimal]) -> Mapping[str, Decimal] | None:
    """Sum ``provider:model`` costs by provider.

    Returns None when any key carries no provider (a record written before
    usage was provider-qualified), rather than a partial breakdown."""

    totals: dict[str, Decimal] = {}
    for key, cost in per_model.items():
        if (parts := split_usage_key(key)) is None:
            return None
        provider, _ = parts
        totals[provider] = totals.get(provider, Decimal(0)) + cost
    return totals


@dataclass(frozen=True)
class Pricing:
    """The one server-wide set of per-model prices.

    ``models`` is keyed either by ``provider:model`` — the price of that model
    when served by that one provider — or by a bare model id, the price of the
    model under any provider. An exact ``provider:model`` entry wins over the
    bare one."""

    currency: str
    models: Mapping[str, ModelPricing]

    def calculate_model(self, model_name: str, usage: UsageInfo) -> Decimal:
        """Calculate one model's exact cost or reject an unknown model.

        ``model_name`` is a ``provider:model`` usage key. A provider-reported
        cost (``usage.extra["reported_cost"]`` — e.g. OpenRouter's usage
        accounting) is authoritative: it is the money actually charged in USD,
        so it wins over token derivation and needs no pricing-table entry for
        the model."""

        if (reported := _reported_cost(usage)) is not None:
            if self.currency != _REPORTED_COST_CURRENCY:
                raise PricingConfigurationError(
                    "Provider-reported costs are denominated in USD, but server pricing "
                    f"uses '{self.currency}'"
                )
            return reported

        model_pricing = self._lookup(model_name)
        if model_pricing is None:
            raise PricingConfigurationError(
                f"Model '{model_name}' is not configured in the server pricing"
            )
        return model_pricing.calculate(usage)

    def covers_model(self, key: str) -> bool:
        """Whether an explicit exact/bare model entry exists (including a zero rate)."""
        return self._lookup(key) is not None

    def _lookup(self, key: str) -> ModelPricing | None:
        if (exact := self.models.get(key)) is not None:
            return exact
        if (parts := split_usage_key(key)) is not None:
            _, model = parts
            return self.models.get(model)
        return None

    def calculate(self, usage_by_model: Mapping[str, UsageInfo]) -> Decimal:
        """Calculate the combined cost of a model-keyed usage snapshot."""

        return sum(
            (
                self.calculate_model(model_name, usage)
                for model_name, usage in usage_by_model.items()
            ),
            start=Decimal(0),
        )


@dataclass(frozen=True)
class TraceCosts:
    """An immutable cost snapshot created with one trace's bound pricing."""

    currency: str
    total: Decimal
    per_model: Mapping[str, Decimal]

    @property
    def per_provider(self) -> Mapping[str, Decimal] | None:
        """Costs summed by provider, or None if a model key names no provider."""

        return costs_per_provider(self.per_model)


@dataclass(frozen=True)
class SessionBudgetDefinition:
    name: str
    value: Decimal
    currency: str
    window: timedelta | None = None

    @property
    def key(self) -> str:
        return self.name


class SessionBudgetLedger:
    """A typed view over one session's durable usage metadata."""

    def __init__(self, data: Mapping[str, object] | None = None) -> None:
        source = data or {}
        self._records: dict[str, dict[str, Any]] = {
            str(trace_id): dict(cast(Mapping[str, Any], record))
            for trace_id, record in cast(Mapping[str, object], source.get("records", {})).items()
        }
        self._baselines: dict[str, dict[str, int]] = {
            str(key): {"sequence": cast(int, cast(Mapping[str, object], value)["sequence"])}
            for key, value in cast(Mapping[str, object], source.get("baselines", {})).items()
        }

    @property
    def latest_sequence(self) -> int:
        return max((int(record["sequence"]) for record in self._records.values()), default=0)

    def initialize(self, budget: SessionBudgetDefinition) -> bool:
        if budget.key in self._baselines:
            return False
        self._baselines[budget.key] = {"sequence": self.latest_sequence}
        return True

    def reset(self, budget: SessionBudgetDefinition) -> None:
        self._baselines[budget.key] = {"sequence": self.latest_sequence}

    def measure(
        self,
        budget: SessionBudgetDefinition,
        *,
        now: datetime | None = None,
        live_cost: Decimal = Decimal(0),
        live_error: str | None = None,
    ) -> Decimal:
        if budget.key not in self._baselines:
            return Decimal(0)

        if live_error:
            raise PricingConfigurationError(live_error)

        baseline = self._baselines[budget.key]["sequence"]
        cutoff = (now or datetime.now(timezone.utc)) - budget.window if budget.window else None
        total = live_cost

        for record in self._records.values():
            if int(record["sequence"]) <= baseline:
                continue
            if cutoff is not None:
                timestamp = datetime.fromisoformat(str(record["timestamp_utc"]))
                if timestamp <= cutoff:
                    continue

            if error := record.get("pricing_error"):
                raise PricingConfigurationError(str(error))

            if cost := cast(Mapping[str, object] | None, record.get("costs")):
                if cost.get("currency") != budget.currency:
                    raise PricingConfigurationError(
                        f"Server pricing changed currency from '{budget.currency}' "
                        f"to '{cost.get('currency')}'"
                    )
                total += Decimal(str(cost["total"]))

        return total

    def exceeded(
        self,
        budget: SessionBudgetDefinition,
        *,
        now: datetime | None = None,
        live_cost: Decimal = Decimal(0),
        live_error: str | None = None,
    ) -> bool:
        return (
            self.measure(
                budget,
                now=now,
                live_cost=live_cost,
                live_error=live_error,
            )
            > budget.value
        )

    def append_record(
        self,
        trace_id: str,
        *,
        timestamp: datetime,
        models: Mapping[str, UsageInfo],
        costs: TraceCosts | None,
        pricing_error: str | None,
    ) -> None:
        self._records[trace_id] = {
            "sequence": self.latest_sequence + 1,
            "timestamp_utc": timestamp.isoformat(),
            "models": {
                model: {
                    "requests": usage.requests,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "extra": dict(usage.extra),
                }
                for model, usage in models.items()
            },
            "costs": (
                {
                    "currency": costs.currency,
                    "total": format(costs.total, "f"),
                    "per_model": {
                        model: format(value, "f") for model, value in costs.per_model.items()
                    },
                }
                if costs is not None
                else None
            ),
            "pricing_error": pricing_error,
        }

    def serialize(self) -> dict[str, JSONSerializable]:
        return cast(
            dict[str, JSONSerializable],
            {
                "records": self._records,
                "baselines": self._baselines,
            },
        )


@dataclass
class _TracePricing:
    """Pricing snapshot and accumulator bound on a trace's first usage report."""

    pricing: Pricing | None
    reported_costs_only: bool = False
    per_model: dict[str, Decimal] = field(default_factory=dict)
    error: str | None = None


class PricingRegistry:
    """Own the server pricing and immutable-per-trace live cost accumulators."""

    def __init__(self, usage_reporter: UsageReporter) -> None:
        self._pricing: Pricing | None = None
        self._explicit_pricing = False
        self._observed_models: dict[str, bool] = {}
        self._budgets: dict[str, SessionBudgetDefinition] = {}
        self._traces: dict[str, _TracePricing] = {}
        usage_reporter.add_listener(self.report_usage)

    def set_pricing(self, pricing: Pricing) -> None:
        """Replace pricing for future traces while preserving in-flight snapshots."""

        if self._budgets and any(
            budget.currency != pricing.currency for budget in self._budgets.values()
        ):
            raise PricingConfigurationError(
                "Server pricing currency cannot change while session budgets are registered"
            )
        self._pricing = pricing
        self._explicit_pricing = True

    @property
    def explicit_pricing(self) -> bool:
        return self._explicit_pricing

    @property
    def observed_models(self) -> Mapping[str, bool]:
        return dict(self._observed_models)

    @property
    def pricing(self) -> Pricing | None:
        return self._pricing

    def register_budget(self, budget: SessionBudgetDefinition) -> None:
        if self._pricing is None:
            # No explicit price table: bind the default empty-USD pricing.
            # Provider-reported costs (OpenRouter, LiteLLM) price against it
            # with no model entries; an operation without a reported cost
            # makes its trace unpriceable, and budget measurement then fails
            # visibly rather than under-counting.
            self._pricing = Pricing(currency=_REPORTED_COST_CURRENCY, models={})
        if budget.currency != self._pricing.currency:
            raise PricingConfigurationError(
                f"Budget currency '{budget.currency}' does not match server pricing currency "
                f"'{self._pricing.currency}'"
            )
        self._budgets[budget.key] = budget

    @property
    def budgets(self) -> Mapping[str, SessionBudgetDefinition]:
        return self._budgets

    def report_usage(self, trace_id: str, model_name: str, usage: UsageInfo) -> None:
        free = _is_free(usage)
        if split_usage_key(model_name) is not None and not free:
            self._observed_models[model_name] = _reported_cost(usage) is not None
        trace = self._traces.get(trace_id)
        if trace is None:
            if self._pricing is None and free:
                # A cache-served usage opening the trace (typically an embedding of
                # the customer's message) must not decide that reported costs
                # can't price it. The first usage that cost something decides.
                return
            if self._pricing is not None:
                trace = _TracePricing(pricing=self._pricing)
            elif _reported_cost(usage) is not None:
                # Built-in gateways such as OpenRouter already report the exact
                # USD charge. Bind an empty USD price table: reported costs need
                # no model entry, while any later unreported operation makes the
                # automatic snapshot incomplete rather than silently free.
                trace = _TracePricing(
                    pricing=Pricing(currency=_REPORTED_COST_CURRENCY, models={}),
                    reported_costs_only=True,
                )
            else:
                trace = _TracePricing(pricing=None)
            self._traces[trace_id] = trace

        if trace.pricing is None or trace.error is not None:
            return

        if trace.reported_costs_only and _reported_cost(usage) is None:
            # A fully-cached usage (all input served from cache, nothing generated)
            # has no provider cost — it's free, not unpriceable — so it contributes
            # zero and must not suppress an otherwise reported-cost trace.
            if free:
                return

            # No explicit price table was configured, so a mixed trace cannot
            # be priced completely. Suppress the partial total without turning
            # ordinary no-pricing operation into a configuration error.
            trace.pricing = None
            trace.per_model.clear()
            return

        try:
            cost = trace.pricing.calculate_model(model_name, usage)
        except PricingConfigurationError as exc:
            trace.error = str(exc)
            return

        trace.per_model[model_name] = trace.per_model.get(model_name, Decimal(0)) + cost

    def live_pricing(self, trace_id: str) -> tuple[Decimal, str | None]:
        trace = self._traces.get(trace_id)
        if trace is None or trace.pricing is None:
            return Decimal(0), None
        return sum(trace.per_model.values(), start=Decimal(0)), trace.error

    def snapshot_trace(self, trace_id: str) -> tuple[TraceCosts | None, str | None]:
        trace = self._traces.get(trace_id)
        if trace is None or trace.pricing is None:
            return None, None
        if trace.error is not None:
            return None, trace.error

        per_model = dict(trace.per_model)
        return (
            TraceCosts(
                currency=trace.pricing.currency,
                total=sum(per_model.values(), start=Decimal(0)),
                per_model=per_model,
            ),
            None,
        )

    def clear_trace(self, trace_id: str) -> None:
        self._traces.pop(trace_id, None)


class SessionBudgetService:
    def __init__(
        self,
        session_store: SessionStore,
        pricing_registry: PricingRegistry,
        tracer: Tracer,
    ) -> None:
        self._session_store = session_store
        self._pricing_registry = pricing_registry
        self._tracer = tracer

    async def measure(
        self,
        budget: SessionBudgetDefinition,
        session_id: SessionId,
        *,
        trace_id: str | None = None,
        now: datetime | None = None,
        workspace: SessionMetadataWorkspace | None = None,
    ) -> Decimal:
        if workspace is not None:
            ledger = self._ledger(workspace.view)
        else:
            session = await self._session_store.read_session(session_id)
            ledger = self._ledger(session.metadata)
        if ledger.initialize(budget):
            if workspace is not None:
                workspace.stage_budget_initialization(budget.key)
            else:
                immediate_workspace = SessionMetadataWorkspace(session, self._session_store)
                immediate_workspace.stage_budget_initialization(budget.key)
                await immediate_workspace.commit()
            return Decimal(0)

        current_trace = trace_id or self._tracer.trace_id
        live_cost, live_error = self._pricing_registry.live_pricing(current_trace)
        return ledger.measure(
            budget,
            now=now,
            live_cost=live_cost,
            live_error=live_error,
        )

    async def exceeded(
        self,
        budget: SessionBudgetDefinition,
        session_id: SessionId,
        *,
        trace_id: str | None = None,
        workspace: SessionMetadataWorkspace | None = None,
    ) -> bool:
        return (
            await self.measure(
                budget,
                session_id,
                trace_id=trace_id,
                workspace=workspace,
            )
        ) > budget.value

    async def reset(
        self,
        budget: SessionBudgetDefinition,
        session_id: SessionId,
        *,
        workspace: SessionMetadataWorkspace | None = None,
    ) -> None:
        if workspace is not None:
            workspace.stage_budget_reset(budget.key)
            return

        session = await self._session_store.read_session(session_id)
        immediate_workspace = SessionMetadataWorkspace(session, self._session_store)
        immediate_workspace.stage_budget_reset(budget.key)
        await immediate_workspace.commit()

    async def record_trace(
        self,
        session_id: SessionId,
        trace_id: str,
        usage: Mapping[str, UsageInfo],
        *,
        timestamp: datetime | None = None,
        workspace: SessionMetadataWorkspace | None = None,
    ) -> Mapping[str, JSONSerializable]:
        costs, pricing_error = self._pricing_registry.snapshot_trace(trace_id)
        timestamp = timestamp or datetime.now(timezone.utc)
        serialized_models = cast(
            dict[str, JSONSerializable],
            {
                model: {
                    "requests": model_usage.requests,
                    "input_tokens": model_usage.input_tokens,
                    "output_tokens": model_usage.output_tokens,
                    "cached_input_tokens": model_usage.cached_input_tokens,
                    "extra": dict(model_usage.extra),
                }
                for model, model_usage in usage.items()
            },
        )
        serialized_costs = cast(
            JSONSerializable,
            (
                {
                    "currency": costs.currency,
                    "total": format(costs.total, "f"),
                    "per_model": {
                        model: format(value, "f") for model, value in costs.per_model.items()
                    },
                }
                if costs is not None
                else None
            ),
        )

        target = workspace
        if target is None:
            session = await self._session_store.read_session(session_id)
            target = SessionMetadataWorkspace(session, self._session_store)

        target.stage_usage_record(
            trace_id=trace_id,
            timestamp_utc=timestamp.isoformat(),
            models=serialized_models,
            costs=serialized_costs,
            pricing_error=pricing_error,
        )

        if workspace is None:
            await target.commit()
            self._pricing_registry.clear_trace(trace_id)

        if pricing_error:
            raise PricingConfigurationError(pricing_error)
        return cast(
            Mapping[str, JSONSerializable],
            target.get_internal(("usage",), {}),
        )

    def acknowledge_trace(self, trace_id: str) -> None:
        """Clear live pricing only after the engine durably commits a trace."""

        self._pricing_registry.clear_trace(trace_id)

    def _ledger(self, metadata: Mapping[str, JSONSerializable]) -> SessionBudgetLedger:
        usage = cast(Mapping[str, object], metadata_utils.internal_value(metadata, ("usage",), {}))
        return SessionBudgetLedger(usage)
