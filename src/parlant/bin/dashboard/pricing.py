"""Informational pricing guidance; it never makes a successful deployment fail."""

from collections.abc import Mapping
from typing import Any

import click

PRICING_DOCS_URL = "https://parlant.io/docs/production/spend-management/#turning-tokens-into-cost"


def echo_pricing_notice(status: Mapping[str, Any]) -> None:
    pricing = status.get("pricing")
    if not isinstance(pricing, Mapping):
        return
    models = pricing.get("models")
    if not isinstance(models, list):
        return
    missing = [
        model["key"]
        for model in models
        if isinstance(model, Mapping)
        and model.get("price") == "missing"
        and isinstance(model.get("key"), str)
    ]
    if missing:
        click.echo("Some models have no price: " + ", ".join(missing) + ".")
        click.echo("Ask your coding agent to add server.set_pricing(...) with your provider rates.")
        click.echo("Prices apply to future turns; earlier unpriced turns remain tokens only.")
        click.echo(f"Pricing guide: {PRICING_DOCS_URL}")
