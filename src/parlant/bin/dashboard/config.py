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

"""Persistent CLI state: interrupted credential refreshes and signed-in workspaces."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

CONFIG_FILENAME = "dashboard.json"


@dataclass(frozen=True)
class PendingRotation:
    """A pending rotation ID associated with a refresh-token fingerprint."""

    rotation_id: str
    credential_fingerprint: str


class ConfigStore(ABC):
    """Durable storage for pending rotations and which workspaces are signed in.

    Pending rotations are kept per credential, since a machine signed in to
    several workspaces holds one refresh token per workspace. The keychain
    cannot list its entries, so the signed-in workspaces are recorded here."""

    @abstractmethod
    def read_pending_rotation(self, credential_fingerprint: str) -> PendingRotation | None: ...

    @abstractmethod
    def write_pending_rotation(self, rotation: PendingRotation) -> None: ...

    @abstractmethod
    def clear_pending_rotation(self, credential_fingerprint: str) -> None: ...

    @abstractmethod
    def signed_in_workspaces(self, host: str) -> list[str]: ...

    @abstractmethod
    def add_signed_in_workspace(self, host: str, workspace: str) -> None: ...

    @abstractmethod
    def remove_signed_in_workspace(self, host: str, workspace: str) -> None: ...

    @abstractmethod
    def active_workspace(self, host: str) -> str | None: ...

    @abstractmethod
    def set_active_workspace(self, host: str, workspace: str | None) -> None: ...


def default_config_path() -> Path:
    return Path(user_config_dir("parlant")) / CONFIG_FILENAME


class FileConfigStore(ConfigStore):
    """Store CLI state in the user configuration directory."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_config_path()

    def read_pending_rotation(self, credential_fingerprint: str) -> PendingRotation | None:
        payload = self._read()
        rotations = payload.get("pending_rotations")
        rotation_id = rotations.get(credential_fingerprint) if isinstance(rotations, dict) else None

        if rotation_id is None:
            # Written by CLIs that kept a single pending rotation.
            legacy = payload.get("pending_rotation")
            if (
                isinstance(legacy, dict)
                and legacy.get("credential_fingerprint") == credential_fingerprint
            ):
                rotation_id = legacy.get("rotation_id")

        if not isinstance(rotation_id, str):
            return None

        return PendingRotation(
            rotation_id=rotation_id,
            credential_fingerprint=credential_fingerprint,
        )

    def write_pending_rotation(self, rotation: PendingRotation) -> None:
        payload = self._read()
        rotations = _mapping(payload, "pending_rotations")
        rotations[rotation.credential_fingerprint] = rotation.rotation_id
        self._write(payload)

    def clear_pending_rotation(self, credential_fingerprint: str) -> None:
        payload = self._read()
        _mapping(payload, "pending_rotations").pop(credential_fingerprint, None)
        legacy = payload.get("pending_rotation")
        if (
            isinstance(legacy, dict)
            and legacy.get("credential_fingerprint") == credential_fingerprint
        ):
            payload.pop("pending_rotation")
        self._write(payload)

    def signed_in_workspaces(self, host: str) -> list[str]:
        workspaces = self._host(self._read(), host).get("workspaces")
        if not isinstance(workspaces, list):
            return []
        return [w for w in workspaces if isinstance(w, str)]

    def add_signed_in_workspace(self, host: str, workspace: str) -> None:
        payload = self._read()
        entry = self._host(payload, host)
        workspaces = self.signed_in_workspaces(host)
        if workspace not in workspaces:
            workspaces.append(workspace)
        entry["workspaces"] = workspaces
        self._write(payload)

    def remove_signed_in_workspace(self, host: str, workspace: str) -> None:
        payload = self._read()
        entry = self._host(payload, host)
        entry["workspaces"] = [w for w in self.signed_in_workspaces(host) if w != workspace]
        self._write(payload)

    def active_workspace(self, host: str) -> str | None:
        active = self._host(self._read(), host).get("active_workspace")
        return active if isinstance(active, str) else None

    def set_active_workspace(self, host: str, workspace: str | None) -> None:
        payload = self._read()
        entry = self._host(payload, host)
        if workspace is None:
            entry.pop("active_workspace", None)
        else:
            entry["active_workspace"] = workspace
        self._write(payload)

    def _host(self, payload: dict[str, Any], host: str) -> dict[str, Any]:
        hosts = _mapping(payload, "hosts")
        entry = hosts.get(host)
        if not isinstance(entry, dict):
            entry = hosts[host] = {}
        return entry

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        # Replace atomically so interrupted writes cannot leave a partial record.
        os.replace(temporary, self._path)


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        value = payload[key] = {}
    return value
