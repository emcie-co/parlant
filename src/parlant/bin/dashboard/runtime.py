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

"""Configuration and dependencies for a CLI invocation."""

from __future__ import annotations

import os
import time
import webbrowser
from dataclasses import dataclass
from typing import Callable

import httpx
from dotenv import find_dotenv, load_dotenv

from parlant.adapters.modules.parlant_dashboard.config import _get_dashboard_base_url
from parlant.bin.dashboard.config import ConfigStore, FileConfigStore
from parlant.bin.dashboard.credentials import CredentialStore, KeyringCredentialStore


REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass
class DashboardRuntime:
    """Dependencies and project selection for a CLI invocation."""

    host: str
    credentials: CredentialStore
    config: ConfigStore
    http: httpx.Client
    open_browser: Callable[[str], bool]
    sleep: Callable[[float], None]
    project: str | None = None
    workspace: str | None = None


def resolve_host() -> str:
    """Resolve the configured dashboard URL."""
    return _get_dashboard_base_url()


def _open_browser(url: str) -> bool:
    try:
        return webbrowser.open(url)
    except Exception:
        return False


def build_runtime() -> DashboardRuntime:
    # Search from the working directory; existing environment variables take precedence.
    load_dotenv(find_dotenv(usecwd=True))

    host = resolve_host()

    return DashboardRuntime(
        host=host,
        credentials=KeyringCredentialStore(),
        config=FileConfigStore(),
        http=httpx.Client(base_url=host, timeout=REQUEST_TIMEOUT_SECONDS),
        open_browser=_open_browser,
        sleep=time.sleep,
        project=os.environ.get("PARLANT_DASHBOARD_PROJECT") or None,
        workspace=os.environ.get("PARLANT_DASHBOARD_WORKSPACE") or None,
    )
