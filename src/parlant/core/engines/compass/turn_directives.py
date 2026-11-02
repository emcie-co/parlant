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

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnDirective:
    """An explicit, non-persistent instruction for the current response."""

    instruction: str
    rationale: str


def format_turn_directives(directives: Sequence[TurnDirective]) -> str:
    """Render directives consistently for the responder and message reviewer."""

    return "\n\n".join(
        (
            f"## Directive {index}\n"
            f"Instruction: {directive.instruction.strip()}\n"
            f"Rationale: {directive.rationale.strip()}"
        )
        for index, directive in enumerate(directives, start=1)
    )
