from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias


ModerationTag: TypeAlias = Literal[
    "jailbreak",
    "harassment",
    "hate",
    "illicit",
    "self-harm",
    "sexual",
    "violence",
]


@dataclass(frozen=True)
class ModerationCheck:
    flagged: bool
    tags: list[ModerationTag]


class ModerationService(ABC):
    @abstractmethod
    async def check(self, content: str) -> ModerationCheck: ...
