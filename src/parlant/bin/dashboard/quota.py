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

"""Validate Backend quota facts without trusting its display text or choosing plan policy."""

from __future__ import annotations

from datetime import datetime, timezone
import re

from parlant.bin.dashboard.errors import CLIQuotaExceeded

_LABELS = {
    "cliTraceReads": "trace-read",
    "cliLogReads": "log-read",
    "cliInspections": "inspection",
}
_MAX_SAFE_INTEGER = 2**53 - 1
_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})"
)


def quota_failure(payload: dict[str, object]) -> CLIQuotaExceeded | None:
    """Only the three known, exhausted workspace allowance shapes are recognized."""
    key = payload.get("quotaKey")
    limit = payload.get("limit")
    message = payload.get("message")
    if (
        type(payload.get("statusCode")) is not int
        or payload["statusCode"] != 429
        or payload.get("error") != "Too Many Requests"
        or payload.get("code") != "cli_quota_exceeded"
        or not isinstance(key, str)
        or key not in _LABELS
        or type(limit) is not int
        or not 0 <= limit <= _MAX_SAFE_INTEGER
        or type(payload.get("remaining")) is not int
        or payload["remaining"] != 0
        or not isinstance(message, str)
        or not 0 < len(message) <= 256
        or "resetAt" not in payload
    ):
        return None

    raw_reset = payload["resetAt"]
    reset = None
    if raw_reset is not None:
        if not isinstance(raw_reset, str) or not _TIMESTAMP.fullmatch(raw_reset):
            return None
        try:
            parsed = datetime.fromisoformat(raw_reset.replace("Z", "+00:00"))
            reset = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except (ValueError, OverflowError):
            return None

    return CLIQuotaExceeded(key, _LABELS[key], limit, reset)
