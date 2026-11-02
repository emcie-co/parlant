# Copyright 2026 Emcie Co Ltd.
# Licensed under the Apache License, Version 2.0.

"""Optional docs attribution; never forward credentials to a custom docs service."""

from __future__ import annotations

import ipaddress
from dataclasses import replace
from urllib.parse import urlsplit

import click
import httpx

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.runtime import DashboardRuntime, build_runtime

DASHBOARD_AUTH_HEADER = "x-parlant-dashboard-authorization"


def credentials_may_be_sent(dashboard: str, endpoint: str) -> bool:
    """Pair each first-party docs origin with only its own platform identity."""
    try:
        platform = urlsplit(dashboard)
        docs = urlsplit(endpoint)
        for parsed in (platform, docs):
            if (
                not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
            ):
                return False
            _ = parsed.port  # Validate before deciding which origin owns credentials.
        if (
            platform.scheme == docs.scheme == "https"
            and platform.port in {None, 443}
            and docs.port in {None, 443}
        ):
            if (platform.hostname, docs.hostname) in {
                ("api.parlant.io", "lighthouse.parlant.io"),
                ("api.emcie.xyz", "lighthouse.emcie.xyz"),
            }:
                return True
        return all(
            parsed.scheme in {"http", "https"} and _loopback(parsed.hostname or "")
            for parsed in (platform, docs)
        )
    except ValueError:
        return False


def _loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def documentation_authorization(
    ctx: click.Context, endpoint: str, workspace: str | None, timeout: float
) -> str | None:
    """Refresh only a selected sign-in; docs remain public if attribution fails.

    The platform derives identity from this access token. Refresh tokens stay on
    the platform/keychain path, never in a Lighthouse request. No catalogued sign-in
    means no keychain probe or sign-in prompt for an ordinary anonymous docs query.
    """
    runtime: DashboardRuntime | None = ctx.find_object(DashboardRuntime)
    owned = runtime is None
    try:
        if runtime is None:
            runtime = build_runtime()
        if not credentials_may_be_sent(runtime.host, endpoint):
            return None
        selected = workspace or runtime.workspace or runtime.config.active_workspace(runtime.host)
        known = runtime.config.signed_in_workspaces(runtime.host)
        if selected is None and not known:
            return None
        if selected is None and len(known) > 1:
            click.echo(
                "No active workspace; use --workspace to attribute this docs query. "
                "Continuing without workspace attribution.",
                err=True,
            )
            return None
        # A per-query selection must not change the user's active workspace.
        request_runtime = replace(runtime, workspace=selected)
        previous_timeout = runtime.http.timeout
        runtime.http.timeout = httpx.Timeout(min(timeout, 3.0))
        try:
            access_token = authenticate(request_runtime)
        finally:
            runtime.http.timeout = previous_timeout
        return f"Bearer {access_token}"
    except (DashboardError, OSError, ValueError):
        # Never print credential-provider errors, headers or private API bodies.
        click.echo(
            "Could not verify the selected sign-in; continuing without workspace attribution.",
            err=True,
        )
        return None
    finally:
        if owned and runtime is not None:
            runtime.http.close()
