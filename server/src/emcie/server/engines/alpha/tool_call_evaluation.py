from typing import Any, Mapping
from emcie.server.base_models import DefaultBaseModel


class ToolCallEvaluation(DefaultBaseModel):
    name: str
    rationale: str
    applicability_score: int
    should_run: bool
    arguments: Mapping[str, Any]
    same_call_is_already_staged: bool


class ToolCallEvaluationsSchema(DefaultBaseModel):
    tool_call_evaluations: list[ToolCallEvaluation]
