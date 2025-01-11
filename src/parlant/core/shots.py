# Copyright 2024 Emcie Co Ltd.
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

from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence


@dataclass
class Shot:
    description: str
    """An explanation of what makes this shot interesting"""


TShot = TypeVar("TShot", bound=Shot)


class ShotCollection(Generic[TShot]):
    def __init__(self, initial_shots: Sequence[TShot]) -> None:
        self._shots: list[TShot] = list(initial_shots)

    async def append(
        self,
        shot: TShot,
    ) -> None:
        self._shots.append(shot)

    async def insert(
        self,
        shot: TShot,
        index: int = 0,
    ) -> None:
        self._shots.insert(index, shot)

    async def list(self) -> Sequence[TShot]:
        return self._shots

    async def remove(
        self,
        shot: TShot,
    ) -> None:
        self._shots.remove(shot)

    async def clear(self) -> None:
        self._shots.clear()
