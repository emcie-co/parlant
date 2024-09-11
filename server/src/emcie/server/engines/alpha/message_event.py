from typing import Optional
from emcie.server.base_models import DefaultBaseModel


class Revision(DefaultBaseModel):
    revision_number: int
    content: str
    rules_followed: list[str]
    rules_broken: list[str]
    followed_all_rules: Optional[bool] = False
    rules_broken_due_to_missing_data: Optional[bool] = False
    missing_data_rationale: Optional[str] = None
    rules_broken_due_to_prioritization: Optional[bool] = False
    prioritization_rationale: Optional[str] = None


class MessageEventSchema(DefaultBaseModel):
    last_message_of_user: str
    produced_reply: bool
    rationale: str
    revisions: list[Revision]
