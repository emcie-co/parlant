from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class UsageInfo:
    input_tokens: int
    output_tokens: int
    extra: Optional[Mapping[str, int]] = None


@dataclass(frozen=True)
class GenerationInfo:
    schema_name: str
    model: str
    duration: float
    usage: UsageInfo
