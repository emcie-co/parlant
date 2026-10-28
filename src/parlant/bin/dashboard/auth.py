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

"""OAuth device authorization and credential refresh."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import click

from parlant.bin.dashboard.client import (
    CLIENT_ID,
    DEVICE_PATH,
    REVOKE_PATH,
    SESSION_PATH,
    TOKEN_PATH,
    decode,
    failure_for,
    oauth_error,
    request,
    require_ok,
)
from parlant.bin.dashboard.config import PendingRotation
from parlant.bin.dashboard.credentials import Credential, credential_key
from parlant.bin.dashboard.errors import (
    CredentialRejected,
    DashboardError,
    credential_rejected,
    no_active_workspace,
    not_signed_in,
)
from parlant.bin.dashboard.runtime import DashboardRuntime

DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
REFRESH_GRANT = "refresh_token"

SCOPE = "workspace.read projects.read projects.create projects.update sessions.read agents.read"

SLOW_DOWN_INCREMENT_SECONDS = 5


@dataclass(frozen=True)
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass(frozen=True)
class Token:
    access_token: str
    refresh_token: str
    session_id: str
    # The workspace a multi-workspace approval issued this sign-in for.
    tenant_id: str | None = None


@dataclass(frozen=True)
class Session:
    user_id: str
    tenant_id: str
    session_id: str
    scope: str


def cli_version() -> str:
    try:
        return metadata.version("parlant")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def _existing_session_ids(runtime: DashboardRuntime) -> list[str]:
    """Offer only this host's saved installation references, never token secrets.

    Backend verifies these hints against the approving browser user, current
    membership/scopes and live grant expiry. Reading them neither refreshes nor
    revokes credentials; a failed keychain read must not prevent a new sign-in.
    """
    try:
        workspaces = runtime.config.signed_in_workspaces(runtime.host)
        active = runtime.config.active_workspace(runtime.host)
        ordered = ([active] if active in workspaces else []) + workspaces
        session_ids: list[str] = []
        for workspace in dict.fromkeys(ordered):
            credential = runtime.credentials.get(credential_key(runtime.host, workspace))
            if credential is None or not re.fullmatch(r"[a-f0-9]{24}", credential.session_id):
                continue
            if credential.session_id not in session_ids:
                session_ids.append(credential.session_id)
            if len(session_ids) == 25:
                break
        return session_ids
    except (DashboardError, OSError, ValueError):
        click.echo(
            "Could not read prior sign-ins. Choose your workspaces on the approval page.",
            err=True,
        )
        return []


def start_device_authorization(runtime: DashboardRuntime) -> DeviceAuthorization:
    existing_session_ids = _existing_session_ids(runtime)
    response = request(
        runtime,
        "POST",
        DEVICE_PATH,
        json_body={
            "client_id": CLIENT_ID,
            "cli_version": cli_version(),
            "scope": SCOPE,
            **({"existing_session_ids": existing_session_ids} if existing_session_ids else {}),
        },
    )

    body = require_ok(response)

    return DeviceAuthorization(
        device_code=str(body["device_code"]),
        user_code=str(body["user_code"]),
        verification_uri=str(body["verification_uri"]),
        verification_uri_complete=str(
            body.get("verification_uri_complete") or body["verification_uri"]
        ),
        expires_in=int(body["expires_in"]),
        interval=int(body["interval"]),
    )


def poll_for_token(runtime: DashboardRuntime, device: DeviceAuthorization) -> list[Token]:
    """Poll for authorization until approval, rejection, or expiration.

    One approval may cover several workspaces: the dashboard then answers with
    one sign-in per workspace, in the order the user chose them. A dashboard
    without multi-workspace approval answers with a single sign-in."""
    interval = device.interval
    waited = 0

    while True:
        response = request(
            runtime,
            "POST",
            TOKEN_PATH,
            json_body={
                "grant_type": DEVICE_GRANT,
                "device_code": device.device_code,
                "client_id": CLIENT_ID,
            },
        )

        if response.is_success:
            return _tokens_from(decode(response))

        error = oauth_error(response)

        if error == "expired_token":
            raise DashboardError(
                "The sign-in code expired before it was approved. "
                "Run `parlant dashboard auth login` again."
            )

        if error == "access_denied":
            raise DashboardError("The sign-in request was declined.")

        if error == "slow_down":
            # RFC 8628 requires the increase to apply to every subsequent poll.
            interval += SLOW_DOWN_INCREMENT_SECONDS
        elif error != "authorization_pending":
            raise failure_for(response)

        if waited + interval > device.expires_in:
            raise DashboardError(
                "The sign-in code expired before it was approved. "
                "Run `parlant dashboard auth login` again."
            )

        runtime.sleep(interval)
        waited += interval


def fingerprint(refresh_token: str) -> str:
    """Identifies a refresh token without storing it outside the keychain."""
    return hashlib.sha256(refresh_token.encode()).hexdigest()


def prepare_rotation(runtime: DashboardRuntime, refresh_token: str) -> str:
    """Persist a rotation ID before refreshing; reuse it after an interrupted request."""
    credential_fingerprint = fingerprint(refresh_token)
    existing = runtime.config.read_pending_rotation(credential_fingerprint)

    if existing is not None:
        return existing.rotation_id

    rotation_id = str(uuid.uuid4())

    runtime.config.write_pending_rotation(
        PendingRotation(
            rotation_id=rotation_id,
            credential_fingerprint=credential_fingerprint,
        )
    )

    return rotation_id


def refresh(runtime: DashboardRuntime, refresh_token: str) -> Token:
    response = request(
        runtime,
        "POST",
        TOKEN_PATH,
        json_body={
            "grant_type": REFRESH_GRANT,
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "rotation_id": prepare_rotation(runtime, refresh_token),
        },
    )

    if response.is_success:
        return _token_from(decode(response))

    if oauth_error(response) == "invalid_grant" or response.status_code in (401, 403):
        raise credential_rejected()

    raise failure_for(response)


def fetch_session(runtime: DashboardRuntime, access_token: str) -> Session:
    response = request(
        runtime,
        "GET",
        SESSION_PATH,
        access_token=access_token,
    )

    body = require_ok(response)

    return Session(
        user_id=str(body["user_id"]),
        tenant_id=str(body["tenant_id"]),
        session_id=str(body["session_id"]),
        scope=str(body.get("scope", "")),
    )


def revoke_session(runtime: DashboardRuntime, credential: Credential) -> None:
    """Attempt remote session revocation without blocking local sign-out."""
    request(
        runtime,
        "POST",
        REVOKE_PATH,
        json_body={
            "refresh_token": credential.refresh_token,
            "client_id": CLIENT_ID,
        },
    )


def sign_in_key(runtime: DashboardRuntime) -> str:
    """The stored sign-in a command uses: `--workspace` (or
    PARLANT_DASHBOARD_WORKSPACE), else the active workspace.

    A sign-in saved before workspaces were tracked is used as it is when no
    workspace is asked for, and identified first when one is."""
    host = runtime.host
    workspaces = runtime.config.signed_in_workspaces(host)
    wanted = runtime.workspace or runtime.config.active_workspace(host)

    if wanted is None and not workspaces:
        if runtime.credentials.get(host) is None:
            raise not_signed_in()
        return credential_key(host)

    if wanted is not None and wanted not in workspaces:
        if identify_legacy_sign_in(runtime) == wanted:
            return credential_key(host, wanted)
        raise not_signed_in(wanted)

    if wanted is None:
        if len(workspaces) > 1:
            raise no_active_workspace(workspaces)
        wanted = workspaces[0]

    return credential_key(host, wanted)


def identify_legacy_sign_in(runtime: DashboardRuntime) -> str | None:
    """Move a sign-in saved under the bare host to its workspace's entry.

    Returns the workspace it belongs to, or None when there is none. It is
    refreshed once to ask the dashboard which workspace it was approved for."""
    host = runtime.host
    if runtime.credentials.get(host) is None:
        return None

    access_token = refresh_stored(runtime, credential_key(host))
    workspace = fetch_session(runtime, access_token).tenant_id
    adopt(runtime, credential_key(host), workspace)
    return workspace


def adopt(runtime: DashboardRuntime, key: str, workspace: str) -> None:
    """Record the sign-in stored at `key` as `workspace`'s sign-in."""
    host = runtime.host
    credential = runtime.credentials.get(key)
    if credential is None:
        return

    target = credential_key(host, workspace)
    if key != target:
        runtime.credentials.save(target, credential)
        runtime.credentials.delete(key)

    runtime.config.add_signed_in_workspace(host, workspace)
    if runtime.config.active_workspace(host) is None:
        runtime.config.set_active_workspace(host, workspace)


def refresh_stored(runtime: DashboardRuntime, key: str) -> str:
    """Refresh the sign-in stored at `key`, clearing it only when rejected."""
    credential = runtime.credentials.get(key)
    workspace = workspace_of(key)

    if credential is None:
        if workspace is not None:
            forget(runtime, workspace)
        raise not_signed_in(workspace)

    try:
        token = refresh(runtime, credential.refresh_token)
    except CredentialRejected:
        runtime.credentials.delete(key)
        if workspace is not None:
            forget(runtime, workspace)
        raise

    runtime.credentials.save(
        key,
        Credential(refresh_token=token.refresh_token, session_id=token.session_id),
    )

    runtime.config.clear_pending_rotation(fingerprint(credential.refresh_token))

    return token.access_token


def workspace_of(key: str) -> str | None:
    _, separator, workspace = key.partition("#workspace=")
    return workspace if separator else None


def forget(runtime: DashboardRuntime, workspace: str) -> None:
    """Drop `workspace` from the signed-in list; with one sign-in left, it becomes active."""
    host = runtime.host
    runtime.config.remove_signed_in_workspace(host, workspace)
    if runtime.config.active_workspace(host) == workspace:
        remaining = runtime.config.signed_in_workspaces(host)
        runtime.config.set_active_workspace(host, remaining[0] if len(remaining) == 1 else None)


def authenticate(runtime: DashboardRuntime) -> str:
    """Refresh the access token for the workspace this command acts in."""
    return refresh_stored(runtime, sign_in_key(runtime))


def _token_from(body: Any) -> Token:
    tenant_id = body.get("tenant_id") if isinstance(body, dict) else None
    return Token(
        access_token=str(body["access_token"]),
        refresh_token=str(body["refresh_token"]),
        session_id=str(body["session_id"]),
        tenant_id=str(tenant_id) if tenant_id else None,
    )


def _tokens_from(body: Any) -> list[Token]:
    if not (isinstance(body, dict) and "grants" in body):
        return [_token_from(body)]

    grants = body["grants"]
    if not isinstance(grants, list) or not grants:
        raise DashboardError(
            "The Parlant dashboard approved the sign-in for no workspace. "
            "Run `parlant dashboard auth login` again and choose at least one."
        )
    return [_token_from(grant) for grant in grants]
