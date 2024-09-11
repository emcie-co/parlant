from dataclasses import dataclass
from typing import Sequence

from emcie.server.core.guidelines import Guideline
from emcie.server.base_models import DefaultBaseModel


@dataclass(frozen=True)
class GuidelineProposition:
    guideline: Guideline
    score: int
    rationale: str


class GuidelinePropositionSchema(DefaultBaseModel):
    predicate_number: int
    predicate: str
    was_already_addressed_or_resolved_according_to_the_record_of_the_interaction: bool
    can_we_safely_presume_to_ascertain_whether_the_predicate_still_applies: bool
    rationale: str
    applies_score: int


class GuidelinePropositionListSchema(DefaultBaseModel):
    checks: Sequence[GuidelinePropositionSchema]
