from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Mapping, NewType, Optional, Sequence, TypeAlias, Union
import hashlib
import nanoid  # type: ignore
from pydantic import BaseModel, ConfigDict
import semver  # type: ignore


class DefaultBaseModel(BaseModel):
    """
    Base class for all Emcie Pydantic models.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
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


class ItemNotFoundError(Exception):
    def __init__(self, item_id: UniqueId, message: Optional[str] = None) -> None:
        super().__init__(f"Item '{item_id}' not found" + (f": {message}" if message else ""))


def generate_id() -> UniqueId:
    return UniqueId(nanoid.generate(size=10))


def md5_checksum(input: str) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(input.encode("utf-8"))

    return md5_hash.hexdigest()


class ProgressReport:
    def __init__(self, progress_callback: Callable[[float], Awaitable[None]]) -> None:
        self._total = 0
        self._current = 0
        self._lock = asyncio.Lock()
        self._progress_callback = progress_callback

    @property
    def percentage(self) -> float:
        if self._total == 0:
            return 0.0
        return self._current / self._total * 100

    async def stretch(self, amount: int) -> None:
        async with self._lock:
            self._total += amount
            await self._progress_callback(self.percentage)

    async def increment(self, amount: int = 1) -> None:
        async with self._lock:
            self._current += amount
            await self._progress_callback(self.percentage)
