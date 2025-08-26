# Copyright 2025 Emcie Co Ltd.
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

from __future__ import annotations
import base64
from collections import defaultdict
from enum import Enum
import asyncio
import hashlib
from typing import Any, Mapping, NewType, Optional, Sequence, TypeAlias, Union
from typing_extensions import Self

import nanoid  # type: ignore
from pydantic import BaseModel, ConfigDict
import semver  # type: ignore


def _without_dto_suffix(obj: Any, *args: Any) -> str:
    if isinstance(obj, str):
        name = obj
        if name.endswith("DTO"):
            return name[:-3]
        return name
    if isinstance(obj, type):
        name = obj.__name__
        if name.endswith("DTO"):
            return name[:-3]
        return name
    else:
        raise Exception("Invalid input to _without_dto_suffix()")


class DefaultBaseModel(BaseModel):
    """
    Base class for all Parlant Pydantic models.
    """

    model_config = ConfigDict(
        validate_default=True,
        model_title_generator=_without_dto_suffix,
    )


JSONSerializable: TypeAlias = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "JSONSerializable"],
    Sequence["JSONSerializable"],
    Optional[str],
    Optional[int],
    Optional[float],
    Optional[bool],
    Optional[None],
    Optional[Mapping[str, "JSONSerializable"]],
    Optional[Sequence["JSONSerializable"]],
]
"""A JSON-serializable value."""

UniqueId = NewType("UniqueId", str)


class Version:
    String = NewType("String", str)

    @staticmethod
    def from_string(version_string: Version.String | str) -> Version:
        result = Version(major=0, minor=0, patch=0)
        result._v = semver.Version.parse(version_string)
        return result

    def __init__(
        self,
        major: int,
        minor: int,
        patch: int,
        prerelease: Optional[str] = None,
    ) -> None:
        self._v = semver.Version(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
        )

    def to_string(self) -> Version.String:
        return Version.String(str(self._v))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._v == other._v

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._v < other._v

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._v > other._v


class ItemNotFoundError(Exception):
    def __init__(self, item_id: UniqueId, message: Optional[str] = None) -> None:
        if message:
            super().__init__(f"{message} (id='{item_id}')")
        else:
            super().__init__(f"Item '{item_id}' not found")


class CancellationSuppressionLatch:
    def __init__(self) -> None:
        self._suppressed = False
        self._task = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        if (
            self._suppressed
            and exc_type is not None
            and issubclass(exc_type, asyncio.CancelledError)
        ):
            return True
        return False

    def enable(self) -> None:
        self._suppressed = True


id_generation_alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class IdGenerator:
    def __init__(self) -> None:
        self._unique_checksums: dict[str, int] = defaultdict(int)

    def _generate_deterministic_id(self, unique_str: str, size: int = 10) -> str:
        h = hashlib.md5(unique_str.encode("utf-8")).digest()
        b64 = base64.urlsafe_b64encode(h).decode()
        id = "".join([c for c in b64 if c in id_generation_alphabet])[:size]

        if len(id) < size:
            raise ValueError(
                f"Generated ID '{id}' is shorter than expected size {size}. "
                "This may indicate an issue with the input string or the ID generation logic. "
                "Please open an issue at https://github.com/emcie-co/parlant"
            )

        return id

    def generate(self, content_checksum: str) -> UniqueId:
        self._unique_checksums[content_checksum] += 1
        unique_str = f"{content_checksum}-{self._unique_checksums[content_checksum]}"

        new_id = self._generate_deterministic_id(unique_str, size=10)
        return UniqueId(new_id)


def generate_id() -> UniqueId:
    return UniqueId(nanoid.generate(size=10, alphabet=id_generation_alphabet))


def md5_checksum(input: str) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(input.encode("utf-8"))

    return md5_hash.hexdigest()


def to_json_dict(d: Mapping[str, Any]) -> Mapping[str, Any]:
    def adapt_value(v: Any) -> Any:
        if isinstance(v, Enum):
            return v.value
        return v

    return {k: adapt_value(v) for k, v in d.items()}
