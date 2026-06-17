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

import pytest

from parlant.core.services.tools.plugins import PluginServer


class FailingPluginServer(PluginServer):
    async def serve(self) -> None:
        raise RuntimeError("bind failed")


class SystemExitingUvicornServer:
    started = False

    def __init__(self, config: object) -> None:
        _ = config

    async def _serve(self) -> None:
        try:
            raise OSError("address already in use")
        except OSError:
            raise SystemExit(1)

    async def serve(self) -> None:
        await self._serve()


async def test_that_plugin_server_reports_background_startup_errors() -> None:
    server = FailingPluginServer([], host="127.0.0.1", port=8818)

    with pytest.raises(RuntimeError, match="bind failed"):
        await server.__aenter__()


async def test_that_plugin_server_wraps_uvicorn_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "parlant.core.services.tools.plugins.uvicorn.Server",
        SystemExitingUvicornServer,
    )

    server = PluginServer([], host="127.0.0.1", port=8818, hosted=True)

    with pytest.raises(RuntimeError) as exc_info:
        await server.serve()

    assert "Failed to start plugin server at http://127.0.0.1:8818" in str(exc_info.value)
    assert "address already in use" in str(exc_info.value)
