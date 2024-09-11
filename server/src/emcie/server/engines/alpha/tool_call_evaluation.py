from typing import Any, Mapping, Optional
from emcie.server.base_models import DefaultBaseModel


class ToolCallEvaluation(DefaultBaseModel):
    name: str
    rationale: str
    applicability_score: int
    should_run: bool
    arguments: Mapping[str, Any]
    same_call_is_already_staged: bool


class ToolCallEvaluationsSchema(DefaultBaseModel):
    last_user_message: Optional[str] = None
    most_recent_user_inquiry_or_need: Optional[str] = None
    most_recent_user_inquiry_or_need_was_already_resolved: Optional[bool] = False
    tool_call_evaluations: list[ToolCallEvaluation]
