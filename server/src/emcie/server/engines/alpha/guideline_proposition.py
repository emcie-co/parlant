# This is a separate module to avoid circular dependencies

from dataclasses import dataclass

from emcie.server.core.guidelines import Guideline


@dataclass(frozen=True)
class GuidelineProposition:
    guideline: Guideline
    score: int
    rationale: str
