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

from decimal import Decimal

import pytest

from parlant.core.nlp.common import UsageInfo
from parlant.core.pricing import (
    ModelPricing,
    Pricing,
    PricingConfigurationError,
    PricingRegistry,
    SessionBudgetDefinition,
    TraceCosts,
)
from parlant.core.tracer import LocalTracer
from parlant.core.usage_reporter import UsageReporter


def test_that_model_pricing_uses_exact_per_million_decimal_rates() -> None:
    pricing = ModelPricing(
        input_tokens=Decimal("3.00"),
        cached_input_tokens=Decimal("0.30"),
        output_tokens=Decimal("15.00"),
        extra={"reasoning_tokens": Decimal("15.00")},
    )

    cost = pricing.calculate(
        UsageInfo(
            input_tokens=10_000,
            cached_input_tokens=8_000,
            output_tokens=500,
            extra={"reasoning_tokens": 100, "provider_metadata": "ignored"},
        )
    )

    assert cost == Decimal("0.0174")
    assert isinstance(cost, Decimal)


def test_that_cached_input_is_not_also_priced_as_uncached_input() -> None:
    pricing = ModelPricing(
        input_tokens=Decimal("1"),
        cached_input_tokens=Decimal("0.1"),
        output_tokens=Decimal("0"),
    )

    assert pricing.calculate(
        UsageInfo(input_tokens=10_000, cached_input_tokens=9_000, output_tokens=0)
    ) == Decimal("0.0019")


def test_that_a_fully_cached_embedding_costs_nothing_with_a_zero_cached_rate() -> None:
    # Embeddings have no discounted provider "cached input" tier — the cache is ours —
    # so an embedding model's cached rate is zero. A fully cached hit reports
    # input_tokens == cached_input_tokens, which is therefore priced at nothing.
    pricing = ModelPricing(
        input_tokens=Decimal("0.02"),
        cached_input_tokens=Decimal("0"),
        output_tokens=Decimal("0"),
    )

    assert pricing.calculate(
        UsageInfo(input_tokens=1_000, cached_input_tokens=1_000, output_tokens=0)
    ) == Decimal("0")


def test_that_a_partially_cached_embedding_bills_only_the_uncached_slice() -> None:
    pricing = ModelPricing(
        input_tokens=Decimal("1"),  # $1 per million
        cached_input_tokens=Decimal("0"),
        output_tokens=Decimal("0"),
    )

    # 1000 total, 600 cached → only the 400 freshly-embedded tokens are billed.
    assert pricing.calculate(
        UsageInfo(input_tokens=1_000, cached_input_tokens=600, output_tokens=0)
    ) == Decimal("0.0004")


def test_that_pricing_aggregates_multiple_models() -> None:
    pricing = Pricing(
        currency="credits",
        models={
            "small": ModelPricing(
                input_tokens=Decimal("1"),
                cached_input_tokens=Decimal("1"),
                output_tokens=Decimal("1"),
            ),
            "large": ModelPricing(
                input_tokens=Decimal("10"),
                cached_input_tokens=Decimal("10"),
                output_tokens=Decimal("10"),
            ),
        },
    )

    cost = pricing.calculate(
        {
            "small": UsageInfo(input_tokens=1_000_000, output_tokens=0),
            "large": UsageInfo(input_tokens=1_000_000, output_tokens=0),
        }
    )

    assert cost == Decimal("11")


def test_that_pricing_rejects_an_unpriced_model() -> None:
    pricing = Pricing(currency="USD", models={})

    with pytest.raises(PricingConfigurationError, match="unpriced-model"):
        pricing.calculate({"unpriced-model": UsageInfo(input_tokens=1, output_tokens=0)})


def test_that_setting_pricing_immediately_activates_cost_reporting() -> None:
    reporter = UsageReporter(LocalTracer())
    registry = PricingRegistry(reporter)
    registry.set_pricing(
        Pricing(
            currency="USD",
            models={
                "model": ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("1"),
                    output_tokens=Decimal("1"),
                )
            },
        )
    )
    registry.report_usage(
        "trace",
        "model",
        UsageInfo(input_tokens=1_000_000, output_tokens=0),
    )

    costs, error = registry.snapshot_trace("trace")

    assert costs is not None
    assert costs.total == Decimal("1")
    assert costs.per_model == {"model": Decimal("1")}
    assert error is None


def test_that_provider_reported_cost_activates_usd_reporting_without_a_price_table() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))

    registry.report_usage(
        "trace",
        "openrouter/model-a",
        UsageInfo(input_tokens=100, output_tokens=10, extra={"reported_cost": 0.001}),
    )
    registry.report_usage(
        "trace",
        "openrouter/model-b",
        UsageInfo(input_tokens=50, output_tokens=5, extra={"reported_cost": 0.002}),
    )

    costs, error = registry.snapshot_trace("trace")

    assert costs is not None
    assert costs.currency == "USD"
    assert costs.total == Decimal("0.003")
    assert costs.per_model == {
        "openrouter/model-a": Decimal("0.001"),
        "openrouter/model-b": Decimal("0.002"),
    }
    assert error is None


def test_that_automatic_reported_cost_does_not_publish_a_partial_mixed_trace() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))

    registry.report_usage(
        "trace",
        "openrouter/reported",
        UsageInfo(input_tokens=100, output_tokens=10, extra={"reported_cost": 0.001}),
    )
    registry.report_usage(
        "trace",
        "other/unpriced",
        UsageInfo(input_tokens=50, output_tokens=5),
    )

    costs, error = registry.snapshot_trace("trace")

    assert costs is None
    assert error is None


def test_that_a_fully_cached_usage_does_not_suppress_a_reported_cost_trace() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))

    registry.report_usage(
        "trace",
        "openrouter/model",
        UsageInfo(input_tokens=100, output_tokens=10, extra={"reported_cost": 0.003}),
    )
    # A cache-served embedding: all input served from cache, nothing generated, no
    # reported cost. It's free (not unpriceable), so the trace total stays intact.
    registry.report_usage(
        "trace",
        "openrouter/embedder",
        UsageInfo(input_tokens=40, output_tokens=0, cached_input_tokens=40),
    )

    costs, error = registry.snapshot_trace("trace")

    assert costs is not None
    assert costs.total == Decimal("0.003")
    assert error is None


def test_that_a_fully_cached_first_usage_does_not_suppress_a_reported_cost_trace() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))

    # The turn opens with a cache-served embedding of the customer's message: free,
    # with no reported cost. It must not decide that the trace cannot be priced.
    registry.report_usage(
        "trace",
        "openrouter/embedder",
        UsageInfo(input_tokens=40, output_tokens=0, cached_input_tokens=40),
    )
    registry.report_usage(
        "trace",
        "openrouter/model-a",
        UsageInfo(input_tokens=100, output_tokens=10, extra={"reported_cost": 0.001}),
    )
    registry.report_usage(
        "trace",
        "openrouter/model-b",
        UsageInfo(input_tokens=50, output_tokens=5, extra={"reported_cost": 0.002}),
    )

    costs, error = registry.snapshot_trace("trace")

    assert costs is not None
    assert costs.currency == "USD"
    assert costs.total == Decimal("0.003")
    assert error is None


def test_that_a_fully_cached_usage_does_not_mark_its_model_as_unreported() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))

    registry.report_usage(
        "trace-1",
        "gateway:embedder",
        UsageInfo(input_tokens=40, output_tokens=0, extra={"reported_cost": 0.0001}),
    )
    # A later cache hit for the same model carries no reported cost because nothing
    # was charged, not because the provider stopped reporting.
    registry.report_usage(
        "trace-2",
        "gateway:embedder",
        UsageInfo(input_tokens=40, output_tokens=0, cached_input_tokens=40),
    )

    assert registry.observed_models == {"gateway:embedder": True}


def test_that_active_pricing_snapshots_cost_and_reports_unknown_models() -> None:
    reporter = UsageReporter(LocalTracer())
    registry = PricingRegistry(reporter)
    registry.set_pricing(
        Pricing(
            currency="USD",
            models={
                "priced": ModelPricing(
                    input_tokens=Decimal("1"),
                    cached_input_tokens=Decimal("1"),
                    output_tokens=Decimal("1"),
                )
            },
        )
    )

    registry.report_usage(
        "priced-trace",
        "priced",
        UsageInfo(input_tokens=2_000_000, output_tokens=0),
    )
    costs, error = registry.snapshot_trace("priced-trace")
    assert costs is not None
    assert costs.currency == "USD"
    assert costs.total == Decimal("2")
    assert costs.per_model == {"priced": Decimal("2")}
    assert error is None

    registry.report_usage(
        "bad-trace",
        "missing",
        UsageInfo(input_tokens=1, output_tokens=0),
    )
    costs, error = registry.snapshot_trace("bad-trace")
    assert costs is None
    assert error is not None and "missing" in error


def test_that_price_changes_do_not_mutate_an_existing_snapshot() -> None:
    reporter = UsageReporter(LocalTracer())
    registry = PricingRegistry(reporter)
    usage = UsageInfo(input_tokens=1_000_000, output_tokens=0)

    registry.set_pricing(
        Pricing(
            "USD",
            {"model": ModelPricing(Decimal("1"), Decimal("1"), Decimal("1"))},
        )
    )
    registry.report_usage("old", "model", usage)
    old_snapshot = registry.snapshot_trace("old")

    registry.set_pricing(
        Pricing(
            "USD",
            {"model": ModelPricing(Decimal("2"), Decimal("2"), Decimal("2"))},
        )
    )
    registry.report_usage("new", "model", usage)

    assert old_snapshot[0] is not None and old_snapshot[0].total == Decimal("1")
    new_snapshot = registry.snapshot_trace("new")[0]
    assert new_snapshot is not None and new_snapshot.total == Decimal("2")


def test_that_an_in_flight_trace_keeps_the_pricing_bound_on_its_first_report() -> None:
    reporter = UsageReporter(LocalTracer())
    registry = PricingRegistry(reporter)
    usage = UsageInfo(input_tokens=1_000_000, output_tokens=0)

    registry.set_pricing(
        Pricing(
            "USD",
            {"model": ModelPricing(Decimal("1"), Decimal("1"), Decimal("1"))},
        )
    )
    registry.report_usage("trace", "model", usage)
    registry.set_pricing(
        Pricing(
            "USD",
            {"model": ModelPricing(Decimal("2"), Decimal("2"), Decimal("2"))},
        )
    )
    registry.report_usage("trace", "model", usage)

    snapshot = registry.snapshot_trace("trace")[0]
    assert snapshot is not None
    assert snapshot.total == Decimal("2")


def test_that_pricing_currency_cannot_change_while_a_budget_is_registered() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))
    registry.set_pricing(Pricing(currency="USD", models={}))
    registry.register_budget(SessionBudgetDefinition("budget", Decimal("1"), "USD"))

    with pytest.raises(PricingConfigurationError, match="currency"):
        registry.set_pricing(Pricing(currency="EUR", models={}))


def test_that_the_documented_anthropic_pricing_example_prices_exactly() -> None:
    # Golden values from Anthropic's prompt-caching documentation example
    # (Opus rates: $5/$25, reads 0.1x, writes 1.25x/2x), normalized to our
    # convention: reads fold into input_tokens; writes ride the extra meters.
    pricing = ModelPricing(
        input_tokens=Decimal("5"),
        cached_input_tokens=Decimal("0.5"),
        output_tokens=Decimal("25"),
        extra={
            "cache_write_5m_input_tokens": Decimal("6.25"),
            "cache_write_1h_input_tokens": Decimal("10"),
        },
    )
    usage = UsageInfo(
        input_tokens=2048 + 1800,  # uncached tail + cache reads
        output_tokens=503,
        cached_input_tokens=1800,
        extra={
            "cache_write_5m_input_tokens": 148,
            "cache_write_1h_input_tokens": 100,
        },
    )

    # 2048x5 + 1800x0.5 + 148x6.25 + 100x10 + 503x25 = 25,640 per-million units.
    assert pricing.calculate(usage) == Decimal("0.02564")


def test_that_a_reported_cost_overrides_token_derivation() -> None:
    pricing = Pricing(
        currency="USD",
        models={
            "m": ModelPricing(
                input_tokens=Decimal("5"),
                cached_input_tokens=Decimal("0.5"),
                output_tokens=Decimal("25"),
            )
        },
    )
    usage = UsageInfo(
        input_tokens=1_000_000,
        output_tokens=0,
        extra={"reported_cost": 0.25},
    )

    # Token derivation would say $5; the provider said $0.25 — money wins.
    assert pricing.calculate_model("m", usage) == Decimal("0.25")


def test_that_a_reported_cost_prices_a_model_absent_from_the_table() -> None:
    pricing = Pricing(currency="USD", models={})
    usage = UsageInfo(
        input_tokens=100,
        output_tokens=10,
        extra={"reported_cost": "0.0000001"},
    )

    assert pricing.calculate_model("unlisted/model", usage) == Decimal("0.0000001")

    # Without a reported cost, the unlisted model still fails visibly.
    with pytest.raises(PricingConfigurationError, match="unlisted/model"):
        pricing.calculate_model("unlisted/model", UsageInfo(input_tokens=100, output_tokens=10))


def test_that_registering_a_budget_without_pricing_binds_the_default_usd_pricing() -> None:
    # Cost-reporting providers (OpenRouter, LiteLLM) need no price table, so a
    # budget must not require one either: registering it binds the default
    # empty-USD pricing, and reported costs price against it directly.
    registry = PricingRegistry(UsageReporter(LocalTracer()))
    registry.register_budget(SessionBudgetDefinition("budget", Decimal("1"), "USD"))

    usage = UsageInfo(
        input_tokens=10,
        output_tokens=5,
        extra={"reported_cost": "0.25"},
    )
    registry.report_usage("trace", "openai/gpt-4o", usage)

    snapshot, error = registry.snapshot_trace("trace")
    assert error is None
    assert snapshot is not None
    assert snapshot.currency == "USD"
    assert snapshot.total == Decimal("0.25")


def test_that_the_default_pricing_fails_visibly_for_unreported_models() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))
    registry.register_budget(SessionBudgetDefinition("budget", Decimal("1"), "USD"))

    registry.report_usage("trace", "unreported-model", UsageInfo(input_tokens=10, output_tokens=0))

    snapshot, error = registry.snapshot_trace("trace")
    assert snapshot is None
    assert error is not None and "unreported-model" in error


def test_that_set_pricing_after_a_default_budget_upgrades_cleanly() -> None:
    registry = PricingRegistry(UsageReporter(LocalTracer()))
    registry.register_budget(SessionBudgetDefinition("budget", Decimal("1"), "USD"))

    registry.set_pricing(
        Pricing("USD", {"model": ModelPricing(Decimal("2"), Decimal("2"), Decimal("2"))})
    )

    with pytest.raises(PricingConfigurationError, match="currency"):
        registry.set_pricing(Pricing(currency="EUR", models={}))


# --- Provider-qualified pricing ----------------------------------------------
#
# Usage keys are "provider:model". A price table may list a model under an exact
# provider, or under the bare model id as the rate for that model wherever it
# is served; the exact entry wins when both exist.


def _flat(rate: str) -> ModelPricing:
    return ModelPricing(
        input_tokens=Decimal(rate),
        cached_input_tokens=Decimal(rate),
        output_tokens=Decimal(rate),
    )


def test_that_a_provider_qualified_price_wins_over_the_bare_model_price() -> None:
    pricing = Pricing(
        currency="USD",
        models={"gpt-5.5": _flat("1"), "azure-eu:gpt-5.5": _flat("3")},
    )
    usage = UsageInfo(input_tokens=1_000_000, output_tokens=0)

    assert pricing.calculate_model("azure-eu:gpt-5.5", usage) == Decimal("3")
    assert pricing.calculate_model("openai:gpt-5.5", usage) == Decimal("1")


def test_that_a_bare_model_price_applies_to_the_model_under_any_provider() -> None:
    pricing = Pricing(currency="USD", models={"claude-haiku-5-5": _flat("2")})
    usage = UsageInfo(input_tokens=500_000, output_tokens=0)

    assert pricing.calculate_model("anthropic:claude-haiku-5-5", usage) == Decimal("1")
    assert pricing.calculate_model("bedrock:claude-haiku-5-5", usage) == Decimal("1")


def test_that_an_unpriced_model_error_names_the_provider_qualified_key() -> None:
    pricing = Pricing(currency="USD", models={"openai:gpt-5.5": _flat("1")})

    with pytest.raises(PricingConfigurationError) as exc_info:
        pricing.calculate_model("azure-eu:gpt-5.5", UsageInfo(input_tokens=1, output_tokens=0))

    assert "azure-eu:gpt-5.5" in str(exc_info.value)


def test_that_the_same_model_is_priced_at_each_providers_own_rate() -> None:
    reporter = UsageReporter(LocalTracer())
    registry = PricingRegistry(reporter)
    registry.set_pricing(
        Pricing(
            currency="USD",
            models={
                "anthropic:claude-sonnet-5-5": _flat("3"),
                "openrouter:anthropic/claude-sonnet-5.5": _flat("4"),
            },
        )
    )
    usage = UsageInfo(input_tokens=1_000_000, output_tokens=0)

    registry.report_usage("trace", "anthropic:claude-sonnet-5-5", usage)
    registry.report_usage("trace", "openrouter:anthropic/claude-sonnet-5.5", usage)

    costs, error = registry.snapshot_trace("trace")
    assert error is None
    assert costs is not None
    assert costs.per_model == {
        "anthropic:claude-sonnet-5-5": Decimal("3"),
        "openrouter:anthropic/claude-sonnet-5.5": Decimal("4"),
    }
    assert costs.total == Decimal("7")


def test_that_trace_costs_total_per_provider() -> None:
    costs = TraceCosts(
        currency="USD",
        total=Decimal("9"),
        per_model={
            "anthropic:claude-sonnet-5-5": Decimal("3"),
            "anthropic:claude-haiku-5-5": Decimal("2"),
            "openai:gpt-5.5": Decimal("4"),
        },
    )

    assert costs.per_provider == {"anthropic": Decimal("5"), "openai": Decimal("4")}


def test_that_per_provider_costs_are_absent_when_a_cost_key_has_no_provider() -> None:
    costs = TraceCosts(
        currency="USD",
        total=Decimal("3"),
        per_model={"anthropic:claude-sonnet-5-5": Decimal("2"), "legacy-model": Decimal("1")},
    )

    assert costs.per_provider is None
