# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from parlant.core.agents import Effort
from parlant.core.common import DefaultBaseModel, JSONSerializable, to_prompt_json
from parlant.core.engines.compass.prompt_builder import EventAdaptationFormat, PromptBuilder
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.compass.context import EngineContext
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.nlp.model_routing import ModelRouter
from parlant.core.sessions import EventKind
from parlant.core.tracer import Tracer


class PromptSections(str, Enum):
    """The compaction prompt's section names — public API for prompt providers.

    See the registry note on the ranker's ``PromptSections`` in
    matching/rule_ranker.py.
    """

    TASK_DESCRIPTION = "compacter.task-description"
    OUTPUT_FORMAT = "compacter.output-format"
    DETAIL_LEVEL = "compacter.detail-level"
    SYSTEM_INSTRUCTIONS = "compacter.system-instructions"


class CompactionDetail(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class CompactionPolicy:
    token_threshold: int
    detail_level: CompactionDetail


class CompactionSchema(DefaultBaseModel):
    summary: str


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    generation_info: GenerationInfo


DEFAULT_COMPACTION_POLICIES: Mapping[Effort, CompactionPolicy] = {
    Effort.MIN: CompactionPolicy(token_threshold=24_000, detail_level=CompactionDetail.LOW),
    Effort.LOW: CompactionPolicy(token_threshold=24_000, detail_level=CompactionDetail.LOW),
    Effort.MEDIUM: CompactionPolicy(token_threshold=32_000, detail_level=CompactionDetail.MEDIUM),
    Effort.HIGH: CompactionPolicy(token_threshold=48_000, detail_level=CompactionDetail.HIGH),
    Effort.MAX: CompactionPolicy(token_threshold=64_000, detail_level=CompactionDetail.HIGH),
}


class Compacter:
    _CACHE_BREAKPOINT = PromptBuilder.INTERACTION_HISTORY_HEADER

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        model_router: ModelRouter,
        prompt_config: PromptConfigurationService,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._compass_tracer = CompassTracer(tracer)
        self._model_router = model_router
        self._prompt_config = prompt_config
        self._policies = dict(DEFAULT_COMPACTION_POLICIES)

    def set_policy(self, policy: Mapping[Effort, CompactionPolicy]) -> None:
        missing_efforts = set(Effort).difference(policy.keys())
        if missing_efforts:
            missing = ", ".join(sorted(e.value for e in missing_efforts))
            raise ValueError(f"Compaction policy is missing effort level(s): {missing}")

        invalid_thresholds = [
            effort.value for effort, p in policy.items() if p.token_threshold <= 0
        ]
        if invalid_thresholds:
            invalid = ", ".join(sorted(invalid_thresholds))
            raise ValueError(f"Compaction policy threshold must be positive for: {invalid}")

        self._policies = dict(policy)

    async def estimate_session_token_count(self, context: EngineContext) -> int:
        """Estimate the token footprint of the loaded interaction history."""
        history_text = self._format_interaction_history(context)

        if not history_text.strip():
            return 0

        tokenizer = await self._model_router.get_estimating_tokenizer(
            context=context,
            route="tokenizer",
            qualifiers=context.state.model_route_qualifiers,
        )
        return await tokenizer.estimate_token_count(history_text)

    async def needs_compaction(self, context: EngineContext) -> bool:
        with self._logger.scope("Compacter"), self._tracer.span("compass.compacter.check"):
            policy = self._policy_for(context)
            token_count = await self.estimate_session_token_count(context)
            result = token_count >= policy.token_threshold

            self._compass_tracer.compaction_checked(
                needed=result,
                effort=context.state.dynamic_effort_level,
                threshold=policy.token_threshold,
                token_count=token_count,
            )
            self._logger.debug(
                "Compaction check: "
                f"effort={context.state.dynamic_effort_level.value}, "
                f"tokens={token_count}, threshold={policy.token_threshold}, "
                f"needed={result}"
            )

            return result

    async def compact(self, context: EngineContext) -> CompactionResult:
        with self._logger.scope("Compacter"), self._tracer.span("compass.compacter.generate"):
            policy = self._policy_for(context)
            generator = await self._model_router.get_schematic_generator(
                context=context,
                route="compaction",
                qualifiers=context.state.model_route_qualifiers,
                schema=CompactionSchema,
            )
            route_reasoning_effort = await self._model_router.resolve_schematic_reasoning_effort(
                context=context,
                route="compaction",
                qualifiers=context.state.model_route_qualifiers,
            )
            inference = await generator.generate(
                prompt=self._build_prompt(context, policy),
                hints={
                    # An explicit route reasoning config wins over the
                    # compaction policy's detail-based default.
                    "reasoning_effort": route_reasoning_effort
                    or self._reasoning_effort_for(policy.detail_level),
                    "cache": {
                        "key": self._cache_key(context),
                        "breakpoint": self._CACHE_BREAKPOINT,
                    },
                },
            )

            return CompactionResult(
                summary=inference.content.summary.strip(),
                generation_info=inference.info,
            )

    def _policy_for(self, context: EngineContext) -> CompactionPolicy:
        effort = context.state.dynamic_effort_level
        if effort not in self._policies:
            raise ValueError(f"No compaction policy configured for effort level: {effort.value}")

        return self._policies[effort]

    def _build_prompt(
        self,
        context: EngineContext,
        policy: CompactionPolicy,
    ) -> PromptBuilder:
        builder = self._prompt_config.create_builder(
            "compaction",
            agent_id=context.agent.id,
            qualifiers=context.state.model_route_qualifiers,
            on_build=lambda prompt: self._logger.trace(f"Compacter prompt:\n{prompt}"),
        )

        builder.add_section(
            name=PromptSections.TASK_DESCRIPTION,
            template="""
# TASK DESCRIPTION

You are compacting a long conversation for a conversational AI agent.
Your task is to summarize the earlier interaction so a future agent turn can continue correctly without seeing every original event.

Do not write a user-facing response.
Do not invent facts.
Do not include hidden chain-of-thought or private reasoning.
Preserve concrete facts, user requests, agent commitments, policy-sensitive details, tool calls and tool results, confirmations, denials, unresolved questions, missing information, and remaining tasks.
Distinguish confirmed facts from assumptions, unknowns, and pending choices.
The summary must supersede any previous session summary.
""",
        )

        builder.add_section(
            name=PromptSections.OUTPUT_FORMAT,
            template="""
# OUTPUT FORMAT

Return your summary using exactly this structure:
```json
{result_structure_text}
```
""",
            props={
                "result_structure_text": self._format_output(),
            },
        )

        builder.add_section(
            name=PromptSections.DETAIL_LEVEL,
            template="""
# COMPACTION DETAIL LEVEL

Detail level: {detail_level}

{detail_instructions}
""",
            props={
                "detail_level": policy.detail_level.value,
                "detail_instructions": self._detail_instructions(policy.detail_level),
            },
        )

        builder.add_section(
            name=PromptSections.SYSTEM_INSTRUCTIONS,
            template="""
# SYSTEM INSTRUCTIONS AND POLICIES

The AI agent must continue to follow these instructions after compaction:

###
{system_instructions}
###
""",
            props={
                "system_instructions": self._build_system_instructions(context),
            },
        )

        if context.state.session_summary.strip():
            builder.add_session_summary(context.state.session_summary)

        builder.add_interaction_history(
            context.interaction.events,
            format=EventAdaptationFormat.JSON,
        )

        return builder

    def _cache_key(self, context: EngineContext) -> str:
        return f"{context.session.id}.compacter"

    def _build_system_instructions(self, context: EngineContext) -> str:
        builder = PromptBuilder()

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_context_variables(context.state.context_variables)
        builder.add_glossary(list(context.state.glossary_terms))
        # The full instruction set: the usable inventory plus the
        # statically-included cross-referential rules, deduped by id — the
        # same invariant as the responder and reviewer sites (in practice
        # statically-included rules are already usable; see
        # matching/common.py: "Static inclusion").
        rendered_rules = {g.id: g for g in context.state.usable_rules}
        for g in context.state.statically_included_rules:
            rendered_rules.setdefault(g.id, g)
        builder.add_system_wide_rules(
            list(rendered_rules.values()),
            context.state.tools_by_rule,
        )

        return builder.build()

    def _format_interaction_history(self, context: EngineContext) -> str:
        return "\n".join(
            PromptBuilder.adapt_event(event, format=EventAdaptationFormat.JSON)
            for event in context.interaction.events
            if event.kind != EventKind.STATUS
        )

    def _format_output(self) -> str:
        result: dict[str, JSONSerializable] = {
            "summary": "A durable, self-contained summary of the interaction history that preserves all facts, tool results, commitments, unresolved tasks, missing information, confirmations, denials, and policy-sensitive state needed to continue the session correctly.",
        }

        return to_prompt_json(result)

    def _detail_instructions(self, detail: CompactionDetail) -> str:
        match detail:
            case CompactionDetail.LOW:
                return (
                    "Compact aggressively. Keep only durable facts, active user goals, "
                    "important tool results, unresolved tasks, and policy-sensitive state."
                )
            case CompactionDetail.MEDIUM:
                return (
                    "Preserve the durable facts from LOW detail plus major decision points, "
                    "confirmations, denials, and enough chronology to explain the current state."
                )
            case CompactionDetail.HIGH:
                return (
                    "Preserve precise facts, identifiers, preferences, confirmations, denials, "
                    "summarized tool calls and results, unresolved ambiguities, policy-sensitive details, "
                    "and the latest state of every active request."
                )

    def _reasoning_effort_for(self, detail: CompactionDetail) -> str:
        match detail:
            case CompactionDetail.LOW:
                return "minimal"
            case CompactionDetail.MEDIUM:
                return "low"
            case CompactionDetail.HIGH:
                return "medium"
