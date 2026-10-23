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

"""The built-in "null" provider: null-object model implementations.

Routing a capability to this provider means "this checkpoint is deliberately
waived" — the route is genuinely configured (which satisfies requirements
like an agent's moderation=on), and the resolved implementation does nothing.

Capabilities are added here as null semantics become meaningful for them;
today that is moderation (a service that never flags).
"""

from typing import Any, Mapping

from parlant.core.nlp.model_routing import ModelCapability, ModelImplementationRegistry
from parlant.core.nlp.moderation import NoModeration


def load_null_models(registry: ModelImplementationRegistry) -> None:
    async def moderation(model: str, raw_args: Mapping[str, Any]) -> NoModeration:
        del model, raw_args
        return NoModeration()

    registry.register(
        "null",
        ModelCapability.MODERATION,
        moderation,
        default="moderation",
    )
