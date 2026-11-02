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

"""The default (open-source) composition binds the port-level implementations.

These tests must exercise the OSS engine — nothing beyond this tree is required
to compose or run it. Alternative bindings are opted into only by a plugin test
package's own conftest. ``compass_engine_bindings`` is that single seam, so it
is what these tests assert on."""

from parlant.core.engines.compass.matcher import Matcher as CompassMatcher
from parlant.core.engines.compass.responder.responder import Responder as CompassResponder
from parlant.core.engines.compass.reviewer import BasicReviewer
from parlant.core.engines.compass.reviewer import Reviewer as CompassReviewer


def test_that_the_default_composition_binds_the_oss_implementations(
    compass_engine_bindings: dict[type, type],
) -> None:
    assert compass_engine_bindings[CompassMatcher] is CompassMatcher
    assert compass_engine_bindings[CompassReviewer] is BasicReviewer
    assert compass_engine_bindings[CompassResponder] is CompassResponder
