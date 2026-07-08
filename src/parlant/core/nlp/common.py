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

from enum import IntEnum
from dataclasses import asdict, dataclass, field
import json
from typing import Mapping
from typing_extensions import Literal, TypeAlias


class ModelSize(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    AUTO = 99


ModelGeneration: TypeAlias = Literal["auto", "stable", "latest"]

ModelType: TypeAlias = Literal["auto", "standard", "reasoning"]


@dataclass(frozen=True)
class UsageInfo:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    extra: Mapping[str, int | float | str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return json.dumps(asdict(self), indent=2)
