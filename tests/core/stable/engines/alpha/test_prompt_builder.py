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

from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.sessions import EventSource

from tests.core.common.utils import create_event_message


def test_that_adapt_event_does_not_escape_non_ascii_characters() -> None:
    event = create_event_message(
        offset=0,
        source=EventSource.CUSTOMER,
        message="Привет, как дела?",
    )

    result = PromptBuilder.adapt_event(event)

    assert "Привет, как дела?" in result
    assert "\\u041f" not in result
