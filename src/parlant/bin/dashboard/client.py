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

"""HTTP requests and event streams for dashboard commands."""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterator, Mapping

import httpx
from httpx_sse import connect_sse

from parlant.bin.dashboard.errors import (
    DashboardError,
    CLIStreamLimitReached,
    credential_rejected,
    forbidden,
    permission_denied,
    platform_failure,
    unreachable,
)
from parlant.bin.dashboard.runtime import REQUEST_TIMEOUT_SECONDS, DashboardRuntime
from parlant.bin.dashboard.quota import quota_failure

DEVICE_PATH = "/v1/cli/auth/device"
TOKEN_PATH = "/v1/cli/auth/token"
SESSION_PATH = "/v1/cli/auth/session"
REVOKE_PATH = "/v1/cli/auth/revoke"
PROJECTS_PATH = "/projects"
WORKSPACES_PATH = "/users/self/workspaces"


def project_path(project_id: str, *segments: str) -> str:
    """A path under one project, e.g. /projects/{id}/status."""
    return "/".join((PROJECTS_PATH, project_id, *segments))


def agents_path(project_id: str, agent_id: str | None = None) -> str:
    return project_path(project_id, "agents", *((agent_id,) if agent_id else ()))


def sessions_path(project_id: str, session_id: str | None = None, *segments: str) -> str:
    return project_path(project_id, "sessions", *((session_id,) if session_id else ()), *segments)


def deployments_path(project_id: str, *segments: str) -> str:
    """A path under a managed project's deployments API."""
    return project_path(project_id, "deployments", *segments)


CLIENT_ID = "parlant-cli"
QUOTA_RESPONSE_MAX_BYTES = 64 * 1024


def stream(
    runtime: DashboardRuntime,
    path: str,
    *,
    access_token: str,
    query: Mapping[str, Any] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield server-sent events and raise on stream errors."""
    headers = {"x-request-id": str(uuid.uuid4()), "authorization": f"Bearer {access_token}"}

    try:
        with connect_sse(
            runtime.http,
            "GET",
            path,
            params=dict(query or {}),
            headers=headers,
            timeout=httpx.Timeout(None, connect=REQUEST_TIMEOUT_SECONDS),
        ) as source:
            if not source.response.is_success:
                if source.response.status_code == 429:
                    raise failure_for(_bounded_quota_response(source.response))
                source.response.read()
                raise failure_for(source.response)

            for event in source.iter_sse():
                name = event.event or "message"

                if name == "error":
                    raise _stream_error(event.data)

                yield name, event.data
    except httpx.HTTPError as error:
        raise unreachable(runtime.host, str(error)) from error


def request(
    runtime: DashboardRuntime,
    method: str,
    path: str,
    *,
    json_body: Mapping[str, Any] | None = None,
    access_token: str | None = None,
    query: Mapping[str, Any] | None = None,
) -> httpx.Response:
    """Send an HTTP request and translate transport errors. Status handling is left to callers."""
    headers: dict[str, str] = {
        "x-request-id": str(uuid.uuid4()),
    }

    if access_token is not None:
        headers["authorization"] = f"Bearer {access_token}"

    try:
        # Inspect status before buffering: error JSON can otherwise grow without
        # a bound, including when an SSE endpoint rejects the request.
        with runtime.http.stream(
            method, path, json=json_body, headers=headers, params=dict(query or {})
        ) as response:
            if response.status_code == 429:
                return _bounded_quota_response(response)
            response.read()
            return response
    except httpx.HTTPError as error:
        raise unreachable(runtime.host, str(error)) from error


def _bounded_quota_response(response: httpx.Response) -> httpx.Response:
    """Buffer at most 64 KiB for 429 recognition and always close the source."""
    content = bytearray()
    try:
        try:
            declared = int(response.headers.get("Content-Length", "0"))
        except ValueError:
            declared = 0
        if declared <= QUOTA_RESPONSE_MAX_BYTES:
            for chunk in response.iter_bytes():
                if len(content) + len(chunk) > QUOTA_RESPONSE_MAX_BYTES:
                    content.clear()
                    break
                content.extend(chunk)
    finally:
        response.close()

    # iter_bytes has already decoded Content-Encoding. Preserve other metadata
    # without decoding twice or retaining the original (possibly wrong) length.
    headers = [
        (key, value)
        for key, value in response.headers.multi_items()
        if key.lower() not in ("content-length", "content-encoding")
    ]
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=bytes(content),
        request=response.request,
        extensions=response.extensions,
    )


def _stream_error(data: str) -> DashboardError:
    """Convert a stream error event to a command failure."""
    if len(data) > QUOTA_RESPONSE_MAX_BYTES:
        return DashboardError("The stream reported an error response that was too large.")
    try:
        body = json.loads(data)
        message = body.get("message") if isinstance(body, dict) else None
        if isinstance(body, dict) and body.get("code") == "cli_stream_limit_reached":
            if isinstance(message, str) and 0 < len(message) <= 256:
                return CLIStreamLimitReached()
            return DashboardError("The stream reported an invalid CLI log-stream limit response.")
    except RecursionError:
        return DashboardError("The stream reported an invalid error response.")
    except ValueError:
        message = None

    return DashboardError(
        message.strip()
        if isinstance(message, str) and message
        else f"The stream reported an error: {data[:200]}"
    )


def decode(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as error:
        raise DashboardError(
            "The Parlant dashboard returned a response this version could not read. "
            "Upgrading Parlant may help."
        ) from error


def server_message(response: httpx.Response) -> str:
    """Extract an error message from the response."""
    try:
        body = response.json()
    except ValueError:
        return ""

    if not isinstance(body, dict):
        return ""

    message = body.get("message")

    return message.strip() if isinstance(message, str) else ""


def failure_for(response: httpx.Response) -> DashboardError:
    """Construct an exception for an unsuccessful response."""
    # Only a 401 means the sign-in itself was refused. A 403 is the dashboard
    # declining this request (role, plan limit, a project not accepting
    # changes); signing in again cannot change it, so say what it was.
    if response.status_code == 401:
        return credential_rejected()

    if response.status_code == 403:
        return _forbidden_failure(response)

    if response.status_code == 429:
        if len(response.content) > QUOTA_RESPONSE_MAX_BYTES:
            return platform_failure(429)
        try:
            body = response.json()
        except (ValueError, RecursionError):
            return platform_failure(429)
        if isinstance(body, dict) and (
            body.get("code") == "cli_quota_exceeded" or "quotaKey" in body
        ):
            # Malformed quota payloads must not fall through to server_message:
            # only validated facts and local text belong in these errors.
            return quota_failure(body) or platform_failure(429)

    return platform_failure(response.status_code, server_message(response), refusal_code(response))


def require_ok(response: httpx.Response) -> Any:
    """Decode a successful response or raise an HTTP error."""
    if response.is_success:
        return decode(response)

    raise failure_for(response)


PERMISSION_DENIED_MESSAGE = "authorization_permission_denied"


def _forbidden_failure(response: httpx.Response) -> DashboardError:
    message = server_message(response)

    if message == PERMISSION_DENIED_MESSAGE:
        return permission_denied(_required_permission(response))

    if message:
        return DashboardError(message, code=refusal_code(response))

    return forbidden()


def _required_permission(response: httpx.Response) -> str | None:
    """`requiredPermission` from a permission denial: one name, or any of several."""
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return None

    required = body.get("requiredPermission") if isinstance(body, dict) else None

    if isinstance(required, str) and required:
        return required

    if isinstance(required, list) and required and all(isinstance(p, str) for p in required):
        return "` or `".join(required)

    return None


def refusal_code(response: httpx.Response) -> str:
    """The `code` of a `{code, message}` refusal, or "" if absent."""
    if len(response.content) > QUOTA_RESPONSE_MAX_BYTES:
        return ""
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return ""

    code = body.get("code") if isinstance(body, dict) else None

    return code if isinstance(code, str) else ""


def oauth_error(response: httpx.Response) -> str:
    """The OAuth `error` code from a 4xx token response, or "" if absent."""
    try:
        body = response.json()
    except ValueError:
        return ""

    if not isinstance(body, dict):
        return ""

    error = body.get("error")

    return error if isinstance(error, str) else ""
