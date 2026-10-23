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

"""Credential storage using the operating system keychain."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from parlant.bin.dashboard.errors import DashboardError

KEYCHAIN_SERVICE = "parlant-dashboard"


@dataclass(frozen=True)
class Credential:
    """A refresh token and its session identifier."""

    refresh_token: str
    session_id: str


def credential_key(host: str, workspace: str | None = None) -> str:
    """Where a sign-in is stored: per workspace, or under the bare host for a
    sign-in saved before workspaces were tracked."""
    return f"{host}#workspace={workspace}" if workspace else host


class CredentialStore(ABC):
    """Storage for signed-in credentials, keyed by `credential_key()`."""

    @abstractmethod
    def get(self, host: str) -> Credential | None:
        """Return the stored credential, or None when signed out."""
        ...

    @abstractmethod
    def save(self, host: str, credential: Credential) -> None:
        """Store the credential, replacing any credential for this host."""
        ...

    @abstractmethod
    def delete(self, host: str) -> None:
        """Delete the stored credential if present."""
        ...


class KeyringCredentialStore(CredentialStore):
    """Store credentials in the OS keychain without a plaintext fallback."""

    def get(self, host: str) -> Credential | None:
        import keyring
        from keyring.errors import KeyringError

        try:
            raw = keyring.get_password(KEYCHAIN_SERVICE, host)
        except KeyringError as error:
            raise _unavailable(error) from error

        if raw is None:
            return None

        try:
            payload = json.loads(raw)
            return Credential(
                refresh_token=str(payload["refresh_token"]),
                session_id=str(payload["session_id"]),
            )
        except (ValueError, KeyError, TypeError):
            return None

    def save(self, host: str, credential: Credential) -> None:
        import keyring
        from keyring.errors import KeyringError

        payload = json.dumps(
            {
                "refresh_token": credential.refresh_token,
                "session_id": credential.session_id,
            }
        )

        try:
            keyring.set_password(KEYCHAIN_SERVICE, host, payload)
        except KeyringError as error:
            raise _unavailable(error) from error

    def delete(self, host: str) -> None:
        import keyring
        from keyring.errors import KeyringError, PasswordDeleteError

        try:
            keyring.delete_password(KEYCHAIN_SERVICE, host)
        except PasswordDeleteError:
            return
        except KeyringError as error:
            raise _unavailable(error) from error


def _unavailable(error: Exception) -> DashboardError:
    return DashboardError(
        "Your system credential store is unavailable, so the sign-in cannot be "
        f"read or written ({error}). On Linux this usually means no Secret "
        "Service (gnome-keyring or kwallet) is running."
    )
