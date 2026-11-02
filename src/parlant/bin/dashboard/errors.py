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

"""User-facing errors for dashboard commands."""

from __future__ import annotations

import json
from typing import IO, Any

import click

from parlant.bin.dashboard.output import json_mode

SIGN_IN_HINT = "Run `parlant dashboard auth login` to sign in."


class DashboardError(click.ClickException):
    """A command failure displayed without a traceback.

    `code` names the failure for programs; with `--json` it is printed as
    `{"error": {"code", "message"}}` on stdout, keeping the exit code."""

    code = "ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code

    def show(self, file: IO[Any] | None = None) -> None:
        if not json_mode():
            super().show(file)
            return
        click.echo(json.dumps({"error": {"code": self.code, "message": self.format_message()}}))


class CLIQuotaExceeded(DashboardError):
    """A workspace allowance refusal with Backend-provided quota and reset facts."""

    exit_code = 7
    code = "CLI_QUOTA_EXCEEDED"
    retryable = False
    remaining = 0

    def __init__(
        self,
        quota_key: str,
        label: str,
        limit: int,
        reset_at: str | None,
    ) -> None:
        self.quota_key = quota_key
        self.limit = limit
        self.reset_at = reset_at
        super().__init__(
            f"Your workspace has reached its CLI {label} allowance. "
            f"Quota: {quota_key}; limit: {limit}; remaining: 0; "
            f"resets at: {reset_at or 'unknown'}."
        )


class CLIStreamLimitReached(DashboardError):
    """The admitted log stream ended at its server-configured duration limit."""

    exit_code = 7
    code = "CLI_STREAM_LIMIT_REACHED"
    retryable = False

    def __init__(self) -> None:
        super().__init__("CLI log-stream time limit reached. Start a new log request to continue.")


def not_signed_in(workspace: str | None = None) -> DashboardError:
    if workspace:
        return DashboardError(
            f"You are not signed in to workspace {workspace}. Run `parlant dashboard auth "
            f"login` and approve it for {workspace}.",
            code="NOT_SIGNED_IN",
        )
    return DashboardError(f"You are not signed in. {SIGN_IN_HINT}", code="NOT_SIGNED_IN")


def no_active_workspace(workspaces: list[str]) -> DashboardError:
    return DashboardError(
        f"You are signed in to several workspaces ({', '.join(workspaces)}) and none is "
        "active. Run `parlant dashboard workspace use <workspace>`, or pass --workspace.",
        code="NO_ACTIVE_WORKSPACE",
    )


def no_project_context() -> DashboardError:
    return DashboardError(
        "No project selected. Pass `--project <id>`, or run "
        "`parlant dashboard project list` to find the project's ID."
    )


def not_managed(project_label: str = "") -> DashboardError:
    subject = f"Project {project_label}" if project_label else "This project"
    return DashboardError(
        f"{subject} is not managed-hosted. Deployment secrets and deployments apply only "
        "to managed projects; self-hosted and local projects run their own server, so "
        "their environment is yours to manage."
    )


def not_connected() -> DashboardError:
    return DashboardError(
        "This project has no connected GitHub repository, so there is nothing to build. "
        "Create a managed project with `parlant dashboard project create --kind managed`."
    )


class CredentialRejected(DashboardError):
    """An authentication failure that requires clearing the stored credential."""

    code = "CREDENTIAL_REJECTED"


def credential_rejected() -> CredentialRejected:
    return CredentialRejected(f"Your sign-in has expired. {SIGN_IN_HINT}")


def permission_denied(permission: str | None) -> DashboardError:
    """A 403 for a role or scope that lacks `permission`. The sign-in stays valid."""
    needed = f"`{permission}`" if permission else "the permission this command needs"
    return DashboardError(
        f"Your role in this workspace doesn't include {needed}. "
        "Ask a workspace Owner or Admin to grant it; signing in again won't help.",
        code="PERMISSION_DENIED",
    )


def forbidden() -> DashboardError:
    """A 403 that carried no explanation. The sign-in stays valid."""
    return DashboardError(
        "The Parlant dashboard refused that request (HTTP 403). Your sign-in is still "
        "valid, so signing in again won't help: check your role in this workspace "
        "and your plan's limits.",
        code="FORBIDDEN",
    )


def platform_failure(status_code: int, detail: str = "", code: str = "") -> DashboardError:
    """`code` is the dashboard's own refusal code, kept for programs."""
    if status_code >= 500:
        if detail:
            return DashboardError(detail, code=code)

        return DashboardError(
            f"The Parlant dashboard could not complete that request (HTTP {status_code}). "
            "This is a problem on the dashboard's side, not with your setup — "
            "try again shortly."
        )

    if detail:
        return DashboardError(detail, code=code)

    return DashboardError(
        f"The Parlant dashboard rejected that request (HTTP {status_code}). "
        "This usually means this version of Parlant and your dashboard disagree "
        "about the request; upgrading Parlant may help."
    )


class Unreachable(DashboardError):
    """The request may or may not have reached the dashboard."""

    code = "UNREACHABLE"


def unreachable(host: str, detail: str) -> Unreachable:
    return Unreachable(f"Could not reach the Parlant dashboard at {host}: {detail}")
