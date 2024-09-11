from typing import Optional
from emcie.server.base_models import DefaultBaseModel


class Revision(DefaultBaseModel):
    revision_number: int
    content: str
    guidelines_followed: Optional[list[str]] = []
    guidelines_broken: Optional[list[str]] = []
    followed_all_guidelines: Optional[bool] = False
    guidelines_broken_due_to_missing_data: Optional[bool] = False
    missing_data_rationale: Optional[str] = None
    guidelines_broken_only_due_to_prioritization: Optional[bool] = False
    prioritization_rationale: Optional[str] = None


class GuidelineEvaluation(DefaultBaseModel):
    number: int
    instruction: str
    evaluation: str
    adds_value: str
    data_available: str


class MessageEventSchema(DefaultBaseModel):
    last_message_of_user: str
    produced_reply: bool
    rationale: str
    revisions: list[Revision]
    evaluations_for_each_of_the_provided_guidelines: list[GuidelineEvaluation]
