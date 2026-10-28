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

"""``parlant dashboard auth`` — signing in and out."""

from __future__ import annotations

import click

from parlant.bin.dashboard.auth import (
    Session,
    authenticate,
    fetch_session,
    forget,
    identify_legacy_sign_in,
    poll_for_token,
    revoke_session,
    start_device_authorization,
)
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import runtime_for
from parlant.bin.dashboard.credentials import Credential, credential_key
from parlant.bin.dashboard.errors import DashboardError, not_signed_in
from parlant.bin.dashboard.output import emit_json, json_mode, json_option, say
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples


@dashboard.group()
def auth() -> None:
    """Sign in to the Parlant dashboard."""


@auth.command(
    "login",
    epilog=examples(
        "parlant dashboard auth login",
        "parlant dashboard auth login --no-browser",
        "parlant dashboard auth login --reauthorize",
    ),
)
@click.option(
    "--reauthorize",
    is_flag=True,
    help="Approve updated permissions. The existing sign-in is kept until the new one succeeds.",
)
@click.option("--no-browser", is_flag=True, help="Print the approval link without opening it.")
@json_option
@click.pass_context
def auth_login(ctx: click.Context, reauthorize: bool, no_browser: bool) -> None:
    """Sign in to a workspace, approving this machine in your browser.

    You approve it for one or more workspaces, and each gets its own sign-in.
    Signing in again adds workspaces, or replaces the sign-ins of those you
    approve again; the first workspace you choose becomes the active one."""
    runtime = runtime_for(ctx)

    device = start_device_authorization(runtime)

    if json_mode():
        emit_json(
            {
                "event": "verification",
                "userCode": device.user_code,
                "verificationUri": device.verification_uri,
                "verificationUriComplete": device.verification_uri_complete,
                "expiresIn": device.expires_in,
            }
        )
    else:
        click.echo("")
        click.echo(f"    Your code:  {device.user_code}")
        click.echo(f"    Approve at: {device.verification_uri_complete}")
        click.echo("")

    if not no_browser and runtime.open_browser(device.verification_uri_complete):
        say("Opened your browser. Waiting for approval...")
    else:
        say("Open the link above to approve this sign-in. Waiting...")

    tokens = poll_for_token(runtime, device)

    # A sign-in from before workspaces were tracked is kept as its own workspace.
    if runtime.credentials.get(runtime.host) is not None:
        try:
            identify_legacy_sign_in(runtime)
        except DashboardError:
            pass

    sessions: list[Session] = []
    replaced: list[tuple[Credential, str]] = []
    for token in tokens:
        session = fetch_session(runtime, token.access_token)
        workspace = session.tenant_id
        key = credential_key(runtime.host, workspace)
        previous = runtime.credentials.get(key)

        runtime.credentials.save(
            key,
            Credential(refresh_token=token.refresh_token, session_id=token.session_id),
        )
        runtime.config.add_signed_in_workspace(runtime.host, workspace)
        sessions.append(session)
        if previous is not None and previous.session_id != token.session_id:
            replaced.append((previous, workspace))

    # The first workspace the user chose becomes the active one.
    workspaces = [session.tenant_id for session in sessions]
    runtime.config.set_active_workspace(runtime.host, workspaces[0])

    if json_mode():
        emit_json({"event": "signed-in", **_status(runtime, sessions[0]), "workspaces": workspaces})
    else:
        click.echo("")
        if len(workspaces) == 1:
            click.echo(f"Signed in to workspace {workspaces[0]}. It is now the active workspace.")
        else:
            click.echo(
                f"Signed in to workspaces {', '.join(workspaces)}. "
                f"{workspaces[0]} is now the active workspace."
            )

    for previous, workspace in replaced:
        try:
            revoke_session(runtime, previous)
        except DashboardError:
            say(
                f"The new sign-in for {workspace} is saved. "
                "Its previous session could not be revoked yet."
            )


@auth.command(
    "status",
    epilog=examples(
        "parlant dashboard auth status",
        "parlant dashboard auth status --json",
    ),
)
@json_option
@click.pass_context
def auth_status(ctx: click.Context) -> None:
    """Show who you are signed in as, and in which workspaces."""
    runtime = runtime_for(ctx)

    _identify_legacy(runtime)
    workspaces = runtime.config.signed_in_workspaces(runtime.host)

    if not workspaces and runtime.credentials.get(runtime.host) is None:
        if json_mode():
            raise not_signed_in()
        click.echo(f"You are not signed in to {runtime.host}.")
        click.echo("Run `parlant dashboard auth login` to sign in.")
        return

    session = fetch_session(runtime, authenticate(runtime))

    if json_mode():
        emit_json(_status(runtime, session))
        return

    click.echo(f"Signed in to {runtime.host}")
    click.echo(f"  workspace: {session.tenant_id}")
    click.echo(f"  user:      {session.user_id}")
    others = [w for w in workspaces if w != session.tenant_id]
    if others:
        click.echo(f"  also signed in to: {', '.join(others)}")


@auth.command(
    "logout",
    epilog=examples(
        "parlant dashboard auth logout",
        "parlant dashboard auth logout --all",
    ),
)
@click.option("--all", "everywhere", is_flag=True, help="Sign out of every workspace.")
@json_option
@click.pass_context
def auth_logout(ctx: click.Context, everywhere: bool) -> None:
    """Sign out of the active workspace (or --workspace) on this machine."""
    runtime = runtime_for(ctx)

    legacy = runtime.credentials.get(runtime.host)
    if legacy is not None:
        try:
            identify_legacy_sign_in(runtime)
        except DashboardError:
            # Signing out must work offline: drop the unidentified sign-in as before.
            try:
                revoke_session(runtime, legacy)
            except DashboardError:
                say("Could not reach the dashboard to end the session remotely.")
            runtime.credentials.delete(runtime.host)
            if not runtime.config.signed_in_workspaces(runtime.host):
                if json_mode():
                    emit_json({"signedIn": False, "signedOut": []})
                else:
                    click.echo("Signed out.")
                return

    workspaces = runtime.config.signed_in_workspaces(runtime.host)
    if everywhere:
        targets = workspaces
    else:
        target = runtime.workspace or runtime.config.active_workspace(runtime.host)
        if target is None and len(workspaces) == 1:
            target = workspaces[0]
        targets = [target] if target in workspaces else []

    if not targets:
        if json_mode():
            emit_json({"signedIn": False, "signedOut": []})
            return
        click.echo("You are not signed in." if not workspaces else "No workspace selected.")
        return

    for workspace in targets:
        key = credential_key(runtime.host, workspace)
        credential = runtime.credentials.get(key)
        if credential is not None:
            try:
                revoke_session(runtime, credential)
            except DashboardError:
                say(f"Could not reach the dashboard to end the {workspace} session remotely.")
            runtime.credentials.delete(key)
        forget(runtime, workspace)

    remaining = runtime.config.signed_in_workspaces(runtime.host)
    if json_mode():
        emit_json({"signedIn": bool(remaining), "signedOut": targets})
        return
    click.echo(f"Signed out of {', '.join(targets)}.")
    active = runtime.config.active_workspace(runtime.host)
    if active:
        click.echo(f"Active workspace: {active}.")
    elif remaining:
        click.echo("Run `parlant dashboard workspace use <workspace>` to choose the active one.")


def _identify_legacy(runtime: DashboardRuntime) -> None:
    if runtime.credentials.get(runtime.host) is not None:
        identify_legacy_sign_in(runtime)


def _status(runtime: DashboardRuntime, session: Session) -> dict[str, object]:
    return {
        "signedIn": True,
        "host": runtime.host,
        "workspace": session.tenant_id,
        "user": {"id": session.user_id},
    }
