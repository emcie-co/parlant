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

"""A managed project's deployments API (`/projects/{id}/deployments`).

Changes (a deploy, a settings or secret save) carry an operation ID and the
configuration revision they were decided against. The platform replays a
repeated operation ID instead of acting twice, so a request whose response was
lost is sent again with the SAME ID; a revision that moved on is re-read and the
change decided again under a NEW ID.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Mapping

import httpx

from parlant.bin.dashboard.client import (
    decode,
    deployments_path,
    failure_for,
    refusal_code,
    request,
    require_ok,
)
from parlant.bin.dashboard.errors import DashboardError, Unreachable
from parlant.bin.dashboard.runtime import DashboardRuntime

SEND_ATTEMPTS = 4

REVISION_ATTEMPTS = 3

OUTCOME_WAIT_SECONDS = 60.0

RETRY_INTERVAL_SECONDS = 2.0


class Refused(DashboardError):
    """A definitive refusal from the deployments API, with its contract code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def overview(runtime: DashboardRuntime, project: str, access_token: str) -> Mapping[str, Any]:
    body = require_ok(request(runtime, "GET", deployments_path(project), access_token=access_token))

    if not isinstance(body, Mapping) or not isinstance(body.get("revision"), str):
        raise DashboardError(
            "The Parlant dashboard returned an unexpected deployment overview. "
            "Upgrading Parlant may help."
        )

    return body


def change(
    runtime: DashboardRuntime,
    project: str,
    access_token: str,
    *,
    method: str,
    segments: tuple[str, ...],
    body_for: Callable[[Mapping[str, Any]], dict[str, Any]],
    accepted_as: str,
) -> Any:
    """Decide a change against the current revision and send it exactly once.

    `accepted_as` names the field of an accepted operation lookup that holds
    what the route itself returns (`attempt` for a deploy, `overview` for a save).
    """
    for _ in range(REVISION_ATTEMPTS):
        current = overview(runtime, project, access_token)
        operation_id = str(uuid.uuid4())
        body = {
            "operationId": operation_id,
            "expectedRevision": current["revision"],
            **body_for(current),
        }

        try:
            return _send_once(runtime, project, access_token, method, segments, body, accepted_as)
        except Refused as refusal:
            if refusal.code != "conflict" or not _revision_moved(
                runtime, project, access_token, current
            ):
                raise

    raise DashboardError(
        "The project's deployment settings kept changing while this command ran. "
        "Try again in a moment."
    )


def _revision_moved(
    runtime: DashboardRuntime, project: str, access_token: str, seen: Mapping[str, Any]
) -> bool:
    return bool(overview(runtime, project, access_token)["revision"] != seen["revision"])


def _send_once(
    runtime: DashboardRuntime,
    project: str,
    access_token: str,
    method: str,
    segments: tuple[str, ...],
    body: dict[str, Any],
    accepted_as: str,
) -> Any:
    last: DashboardError | None = None

    for attempt in range(SEND_ATTEMPTS):
        if attempt:
            runtime.sleep(RETRY_INTERVAL_SECONDS)

        try:
            response = request(
                runtime,
                method,
                deployments_path(project, *segments),
                json_body=body,
                access_token=access_token,
            )
        except Unreachable as error:
            last = error
            continue

        if response.is_success:
            return decode(response)

        code = refusal_code(response)

        if code == "unknown-outcome":
            return _await_outcome(runtime, project, access_token, body["operationId"], accepted_as)

        if _may_not_have_arrived(response, code):
            last = failure_for(response)
            continue

        raise _refused(response, code)

    assert last is not None
    raise last


def _may_not_have_arrived(response: httpx.Response, code: str) -> bool:
    return response.status_code >= 500 and response.status_code != 501 or code == "network"


def _refused(response: httpx.Response, code: str) -> DashboardError:
    failure = failure_for(response)

    return Refused(code, str(failure.message)) if code else failure


def _await_outcome(
    runtime: DashboardRuntime,
    project: str,
    access_token: str,
    operation_id: str,
    accepted_as: str,
) -> Any:
    deadline = time.monotonic() + OUTCOME_WAIT_SECONDS

    while True:
        result = require_ok(
            request(
                runtime,
                "GET",
                deployments_path(project, "operations", operation_id),
                access_token=access_token,
            )
        )
        status = result.get("status") if isinstance(result, Mapping) else None

        if status == "failed":
            error = result.get("error")
            error = error if isinstance(error, Mapping) else {}
            raise Refused(
                str(error.get("code", "")), str(error.get("message", "The request failed."))
            )

        if status in ("not-found", "expired"):
            raise DashboardError("That request did not take effect. Run the command again.")

        if status != "pending":
            if not isinstance(result, Mapping) or accepted_as not in result:
                raise DashboardError(
                    "The Parlant dashboard returned an unexpected operation result. "
                    "Upgrading Parlant may help."
                )
            return result[accepted_as]

        if time.monotonic() >= deadline:
            raise DashboardError(
                f"The platform is still deciding request {operation_id}. Check again shortly; "
                "running the command again is safe."
            )

        runtime.sleep(RETRY_INTERVAL_SECONDS)
