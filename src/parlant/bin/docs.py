# Copyright 2026 Emcie Co Ltd.
# Licensed under the Apache License, Version 2.0.

"""Ask Lighthouse for current, source-backed Parlant documentation over HTTP.

This client does not guess answers or fall back to a different knowledge source.
The production service is the default; PARLANT_DOCS_URL or --endpoint can select
a development or local instance.
"""

from __future__ import annotations

import ipaddress
import json
from importlib import metadata
from typing import Literal
from urllib.parse import urlsplit

import click
import httpx
from pydantic import BaseModel, Field, ValidationError

from parlant.bin.docs_attribution import DASHBOARD_AUTH_HEADER, documentation_authorization
from parlant.bin.help import examples


class Citation(BaseModel):
    document_id: str
    title: str
    url: str


class DocumentationAnswer(BaseModel):
    request_id: str = Field(min_length=1)
    status: Literal["answered", "insufficient_documentation"]
    answer: str = Field(min_length=1)
    citations: list[Citation]
    warnings: list[str]
    snapshot_revision: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)
    model: str = Field(min_length=1)
    usage: dict[str, int]
    duration_ms: int = Field(ge=0)


def _endpoint(value: str | None) -> str:
    if not value:
        raise click.ClickException(
            "Set PARLANT_DOCS_URL or pass --endpoint for the Lighthouse documentation service."
        )
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        local = hostname == "localhost"
        if not local:
            try:
                local = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                local = False
        valid = (
            bool(hostname)
            and (parsed.scheme == "https" or (parsed.scheme == "http" and local))
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )
        if not valid:
            raise ValueError("invalid endpoint")
    except ValueError:
        raise click.ClickException(
            "The docs endpoint must use HTTPS (HTTP is allowed for localhost) "
            "and contain no credentials, query string or fragment."
        ) from None
    return value.rstrip("/")


@click.group()
def docs() -> None:
    """Consult current Parlant documentation through Lighthouse."""


@docs.command(
    "query",
    epilog=examples(
        'parlant docs query "How do I add a tool to an agent?"',
        'parlant docs query "Why is my policy not matching?" --context "$(cat domains/orders/domain.py)"',
        'parlant docs query "Should this be a policy or an observation?" --context "The rule is: never quote a price to an unpaid account."',
        'parlant docs query "How do I test agent behavior?" --json',
    ),
)
@click.argument("question")
@click.option(
    "--context", default="", help="Relevant project context; omit credentials and secrets."
)
@click.option(
    "--endpoint",
    envvar="PARLANT_DOCS_URL",
    default="https://lighthouse.parlant.io",
    show_default=True,
    help="Lighthouse service base URL.",
)
@click.option(
    "--workspace",
    envvar="PARLANT_DASHBOARD_WORKSPACE",
    default=None,
    help="Attribute the query to this signed-in workspace without changing the active workspace.",
)
@click.option("--json", "as_json", is_flag=True, help="Return the complete structured response.")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True, max=180),
    default=120.0,
    show_default=True,
    help="Maximum HTTP request time in seconds.",
)
@click.pass_context
def query_docs(
    ctx: click.Context,
    question: str,
    context: str,
    endpoint: str | None,
    workspace: str | None,
    as_json: bool,
    timeout: float,
) -> None:
    """Ask a question about Parlant and get a source-backed answer.

    QUESTION is the question itself, in plain English. Add --context with what
    you are building, what you have tried, and the code it concerns: the answer
    is only as specific as the context you give it.

    Questions and context are sent to Lighthouse and its Google model provider.
    Lighthouse retains secret-redacted request/answer records for staff review.
    Do not include credentials, customer data or anything from your .env.

    When signed in to the matching platform, the query record includes your
    verified user/workspace names and IDs for staff housekeeping. Custom docs
    endpoints receive no dashboard credentials. Sign-in is never required.
    """
    if not question.strip():
        raise click.ClickException("The documentation question must not be empty.")
    if len(question) > 12_000 or len(context) > 60_000:
        raise click.ClickException("Question or context exceeds the documentation service limit.")
    host = _endpoint(endpoint)
    authorization = documentation_authorization(ctx, host, workspace, timeout)
    try:
        sdk_version = metadata.version("parlant")
    except metadata.PackageNotFoundError:
        sdk_version = "unknown"
    headers = {"User-Agent": f"parlant-cli/{sdk_version}"}
    if authorization is not None:
        headers[DASHBOARD_AUTH_HEADER] = authorization
    try:
        # No automatic retries: an upstream timeout may follow completed model work.
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.post(
                f"{host}/v1/query",
                json={
                    "query": question,
                    "context": context,
                    "sdk_version": sdk_version,
                    "client": f"parlant-cli/{sdk_version}",
                },
                headers=headers,
            )
    except httpx.TimeoutException:
        raise click.ClickException(
            "The documentation request timed out. The service may still have processed it."
        ) from None
    except httpx.HTTPError:
        raise click.ClickException("Could not connect to the documentation service.") from None
    if response.status_code != 200:
        raise click.ClickException(
            f"The documentation service returned HTTP {response.status_code}. "
            "For 429, wait before retrying; for other failures, check the endpoint or try later."
        )
    try:
        answer = DocumentationAnswer.model_validate_json(response.content)
        if answer.status == "answered" and not answer.citations:
            raise ValueError("answered response has no sources")
    except (ValidationError, ValueError):
        raise click.ClickException(
            "The documentation service returned an invalid response."
        ) from None
    if as_json:
        click.echo(json.dumps(answer.model_dump(), ensure_ascii=False))
        return
    click.echo(answer.answer)
    if answer.citations:
        click.echo("\nSources:")
        for citation in answer.citations:
            click.echo(f"- {citation.title}: {citation.url}")
    for warning in answer.warnings:
        click.echo(f"Warning: {warning}", err=True)
