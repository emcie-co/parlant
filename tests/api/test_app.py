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

from fastapi import status
import httpx


async def test_health_check_endpoint(async_client: httpx.AsyncClient) -> None:
    response = await async_client.get("/healthz")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


async def test_that_404_error_responses_include_cors_headers(
    async_client: httpx.AsyncClient,
) -> None:
    """CORS headers must be present on error responses for cross-origin clients."""
    response = await async_client.get(
        "/agents/nonexistent-agent-id",
        headers={"Origin": "http://localhost:3000"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "access-control-allow-origin" in response.headers
