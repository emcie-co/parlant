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

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from parlant.core.common import JSONSerializable
from parlant.core.rules import Rule


@dataclass(frozen=True)
class RuleEvaluation:
    """A provisional judgment about a rule's relevance to the current context."""

    rule: Rule
    reasoning: str
    is_relevant: bool = True
    score: float | None = None
    highlights: Sequence[str] = field(default=())
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)

    @property
    def guideline(self) -> Rule:
        return self.rule

    def __hash__(self) -> int:
        return hash(f"{self.rule.id}_{self.reasoning}")
