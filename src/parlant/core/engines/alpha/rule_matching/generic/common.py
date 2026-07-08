from parlant.core.engines.alpha.guideline_matching.generic.common import (
    GuidelineInternalRepresentation,
    dump_guideline,
    escape_json_string,
    internal_representation,
)

RuleInternalRepresentation = GuidelineInternalRepresentation
dump_rule = dump_guideline

__all__ = [
    "RuleInternalRepresentation",
    "dump_rule",
    "escape_json_string",
    "internal_representation",
]
