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

"""The tool-selection port: which tools should the response model receive?

Tool selection is deliberately separate from rule discovery. Rules are
discovered into a sticky session working set; tools have no corresponding
admission, eviction, or readmission lifecycle. Instead, a selector prepares the
available tool inventory and then chooses the bounded catalog exposed for the
current response.

The two phases let preparation (including relevance scoring) run concurrently
with rule matching. Selection runs afterward so it can prioritize tools enabled
by current-turn evaluations and by rules already discovered for the session.
"""

from abc import ABC, abstractmethod

from parlant.core.engines.compass.context import EngineContext


class ToolSelector(ABC):
    """Prepares and selects the tool catalog exposed for a Compass response.

    ``prepare`` populates the tool inventory and any selector-specific scores
    needed by ``select``. ``select`` consumes the completed rule-matching state
    and writes the final matched and available tool catalogs to the response
    state. Calling ``select`` again after a tool step must not require rescoring
    an otherwise unchanged conversation.
    """

    @abstractmethod
    async def prepare(self, context: EngineContext) -> None: ...

    @abstractmethod
    async def select(self, context: EngineContext) -> None: ...
