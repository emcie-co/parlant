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

from typing import Any

from parlant.core.retrievers import RetrieverRegistry
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


class RetrieverTunnelHandlers:
    def __init__(self, registry: RetrieverRegistry) -> None:
        self._registry = registry

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(unary={"retrievers.list": self.list}, streams={})

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        return {
            "retrievers": [
                {
                    "id": registration.id,
                    "name": registration.name,
                    "title": registration.title,
                    "description": registration.description,
                    "owner": {"kind": registration.owner.kind, "id": registration.owner.id},
                }
                for registration in self._registry.list()
            ]
        }
