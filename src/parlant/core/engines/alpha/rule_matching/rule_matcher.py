from dataclasses import dataclass
from typing import Sequence

from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
    GuidelineMatchingStrategyResolver as RuleMatchingStrategyResolver,
    ResponseAnalysisBatch,
)
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.journeys import Journey
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.rules import Rule


@dataclass(frozen=True)
class RuleMatchingResult:
    total_duration: float
    batch_count: int
    batch_generations: Sequence[GenerationInfo]
    batches: Sequence[Sequence[RuleMatch]]
    matched: Sequence[RuleMatch]
    ruled_out: Sequence[RuleMatch]


def _to_rule_match(match: GuidelineMatch) -> RuleMatch:
    return RuleMatch(
        rule=match.guideline,
        rationale=match.rationale,
        metadata=match.metadata,
    )


class RuleMatcher(GuidelineMatcher):
    async def match_rules(
        self,
        context: EngineContext,
        active_journeys: Sequence[Journey],
        rules: Sequence[Rule],
    ) -> RuleMatchingResult:
        result = await self.match_guidelines(
            context=context,
            active_journeys=active_journeys,
            guidelines=rules,
        )

        return RuleMatchingResult(
            total_duration=result.total_duration,
            batch_count=result.batch_count,
            batch_generations=result.batch_generations,
            batches=[[_to_rule_match(m) for m in batch] for batch in result.batches],
            matched=[_to_rule_match(m) for m in result.matched],
            ruled_out=[_to_rule_match(m) for m in result.ruled_out],
        )


__all__ = [
    "ResponseAnalysisBatch",
    "RuleMatcher",
    "RuleMatchingResult",
    "RuleMatchingStrategyResolver",
]
