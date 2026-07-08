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

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from lagom import Container
from pytest import fixture

from parlant.core.engines.compass.matching.rule_recaller import (
    RuleRecaller,
    _LogisticModel,
)
from parlant.core.engines.compass.response_state import ResponseState
from parlant.core.engines.engine_context import EngineContext
from parlant.core.rules import Rule, RuleContent
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from parlant.core.agents import Agent

from tests.core.stable.engines.compass.matching.utils import (
    RecordedEvent,
    RecordedSpan,
    RecordingTracer,
    create_agent,
    create_engine_context,
    create_rule,
)


@fixture
def recaller(container: Container) -> RuleRecaller:
    return container[RuleRecaller]


def test_that_a_rule_recaller_can_be_created(recaller: RuleRecaller) -> None:
    assert recaller is not None


def test_that_the_logistic_model_separates_a_linearly_separable_set() -> None:
    rng = np.random.default_rng(0)
    positives = rng.normal(loc=[2.0, 2.0], scale=0.3, size=(30, 2))
    negatives = rng.normal(loc=[-2.0, -2.0], scale=0.3, size=(30, 2))
    features = np.vstack([positives, negatives])
    labels = np.array([1] * 30 + [0] * 30)

    model = _LogisticModel.fit(features, labels, C=0.5)

    # Held-out points on each side fall on the correct side of the boundary.
    assert model.decision(np.array([[2.0, 2.0]]))[0] > 0.0
    assert model.decision(np.array([[-2.0, -2.0]]))[0] < 0.0
    # The set is separable, so every positive outscores every negative.
    assert model.decision(positives).min() > model.decision(negatives).max()


def test_that_the_logistic_model_balances_class_weights() -> None:
    # One lone positive against many negatives: balanced weighting must keep it
    # from being drowned out (an unweighted fit would put it below the boundary).
    rng = np.random.default_rng(1)
    positives = np.array([[3.0, 0.0]])
    negatives = rng.normal(loc=[-1.0, 0.0], scale=0.5, size=(100, 2))
    features = np.vstack([positives, negatives])
    labels = np.array([1] + [0] * 100)

    model = _LogisticModel.fit(features, labels, C=0.5, class_weight="balanced")

    assert model.decision(positives)[0] > 0.0


class _FakeTokenizer(EstimatingTokenizer):
    async def estimate_token_count(self, prompt: str) -> int:
        return len(prompt.split())


class _FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str], hints: Mapping[str, Any] = {}) -> EmbeddingResult:
        self.embed_calls.append(texts)
        return EmbeddingResult(vectors=[self._vector_for_text(text) for text in texts])

    @property
    def id(self) -> str:
        return "fake-radar-embedder"

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return _FakeTokenizer()

    @property
    def dimensions(self) -> int:
        return 2

    def _vector_for_text(self, text: str) -> list[float]:
        lowered = text.lower()

        if any(term in lowered for term in ("refund", "money back", "money")):
            return [1.0, 0.0]

        if any(term in lowered for term in ("hours", "open")):
            return [-1.0, 0.0]

        if any(term in lowered for term in ("package", "shipping", "delivery")):
            return [0.0, 1.0]

        return [0.0, -1.0]


class _FakeNLPService:
    def __init__(self, embedder: _FakeEmbedder) -> None:
        self._embedder = embedder

    async def get_embedder(self, hints: Mapping[str, Any] = {}) -> _FakeEmbedder:
        return self._embedder


class _FakeEmbeddingCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[type[Embedder], tuple[str, ...]], EmbeddingResult] = {}

    async def get(
        self, embedder_type: type[Embedder], texts: list[str], hints: Mapping[str, Any] = {}
    ) -> EmbeddingResult | None:
        return self._entries.get((embedder_type, tuple(texts)))

    async def set(
        self,
        embedder_type: type[Embedder],
        texts: list[str],
        vectors: list[list[float]],
        hints: Mapping[str, Any] = {},
    ) -> None:
        self._entries[(embedder_type, tuple(texts))] = EmbeddingResult(vectors=vectors)


def _radar_recaller(
    embedder: _FakeEmbedder | None = None,
    embedding_cache: _FakeEmbeddingCache | None = None,
    tracer: LocalTracer | None = None,
    **kwargs: Any,
) -> RuleRecaller:
    return RuleRecaller(
        nlp_service=_FakeNLPService(embedder or _FakeEmbedder()),  # type: ignore[arg-type]
        tracer=tracer or create_engine_context(conversation=[]).tracer,
        embedding_cache=embedding_cache or _FakeEmbeddingCache(),  # type: ignore[arg-type]
        **kwargs,
    )


def _context(
    conversation: list[tuple[EventSource, str]],
    agent: Agent | None = None,
) -> EngineContext[Any]:
    context = create_engine_context(conversation=conversation, agent=agent)
    context.state = ResponseState()
    return context


def _create_sample_rules() -> dict[str, Rule]:
    refund = create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        groups=[],
    )
    hours = create_rule(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        groups=[],
    )
    shipping = create_rule(
        condition="the customer asks where their package is",
        action="share the shipping status",
        groups=[],
    )

    return {"refund": refund, "hours": hours, "shipping": shipping}


async def test_that_the_recaller_recalls_the_discriminating_policy() -> None:
    recaller = _radar_recaller(recall_margin=0.0)
    rules = _create_sample_rules()
    available = list(rules.values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    result = await recaller.discover(context, available)

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}

    assert result.duration >= 0.0
    assert len(result.discovered_rules) == 3
    assert relevance_by_id[rules["refund"].id]
    assert not relevance_by_id[rules["hours"].id]
    assert not relevance_by_id[rules["shipping"].id]


async def test_that_recaller_records_span_and_per_rule_result_events() -> None:
    span_tracer = RecordingTracer()
    context_tracer = RecordingTracer()
    recaller = _radar_recaller(tracer=span_tracer)
    rule = create_rule(
        condition="customer wants a refund",
        action="start the refund flow",
        title="Refund Flow",
    )
    context = replace(
        _context([(EventSource.CUSTOMER, "I want my money back")]),
        tracer=context_tracer,
    )

    await recaller.discover(context, [rule])

    assert span_tracer.started_spans == [RecordedSpan(name="match.rule.recall", attributes={})]
    assert context_tracer.events == [
        RecordedEvent(
            name="matched.recall.yes",
            attributes={
                "rule_id": str(rule.id),
                "title": "Refund Flow",
                "score": 0.0,
            },
            span_id="<main>",
        )
    ]


async def test_that_the_recaller_stays_sticky_across_user_turns() -> None:
    # The refund-relevant turn is earlier in the conversation; max-over-turns must
    # keep the refund policy relevant even though the latest turn is off-topic.
    recaller = _radar_recaller()
    rules = _create_sample_rules()
    available = list(rules.values())
    context = _context(
        [
            (EventSource.CUSTOMER, "I want my money back"),
            (EventSource.AI_AGENT, "Let me help."),
            (EventSource.CUSTOMER, "what are your opening hours?"),
        ]
    )

    result = await recaller.discover(context, available)

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}

    assert relevance_by_id[rules["refund"].id]


async def test_that_the_recaller_embeds_policy_signals() -> None:
    recaller = _radar_recaller()
    refund = replace(
        create_rule(
            condition="the customer asks for account support",
            action="start the account support flow",
            groups=[],
        ),
        signals=["I want my money back"],
    )
    hours = create_rule(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        groups=[],
    )
    shipping = create_rule(
        condition="the customer asks where their package is",
        action="share the shipping status",
        groups=[],
    )

    context = _context([(EventSource.CUSTOMER, "I want my money back please")])

    result = await recaller.discover(context, [refund, hours, shipping])

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}
    assert relevance_by_id[refund.id]


async def test_that_the_recaller_trains_against_policy_anti_signals() -> None:
    recaller = _radar_recaller()
    refund = replace(
        create_rule(
            condition="the customer asks for account support",
            action="start the account support flow",
            groups=[],
        ),
        signals=["I want my money back"],
        anti_signals=["what are your opening hours"],
    )
    hours = create_rule(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        groups=[],
    )

    context = _context([(EventSource.CUSTOMER, "what are your opening hours?")])

    result = await recaller.discover(context, [refund, hours])

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}
    assert not relevance_by_id[refund.id]
    assert relevance_by_id[hours.id]


async def test_that_a_single_candidate_rule_with_anti_signals_is_scored() -> None:
    recaller = _radar_recaller()
    refund = replace(
        create_rule(
            condition="the customer wants a refund",
            action="start the refund flow",
            groups=[],
        ),
        signals=["I want my money back"],
        anti_signals=["what are your opening hours"],
    )
    context = _context([(EventSource.CUSTOMER, "what are your opening hours?")])

    result = await recaller.discover(context, [refund])

    assert result.discovered_rules
    assert not result.discovered_rules[0].is_relevant


def test_that_the_recaller_formats_description_only_policy_rule() -> None:
    recaller = _radar_recaller()
    rule = replace(
        create_rule(condition="", action=None, groups=[]),
        title="Refund Policy",
        content=RuleContent(
            condition="",
            action=None,
            description="Refunds are allowed within 30 days.",
        ),
    )

    assert recaller._rule_embedding_content(rule) == (
        "# Refund Policy\n\nRefunds are allowed within 30 days."
    )


def test_that_the_recaller_embeds_policy_rule_text_and_signals() -> None:
    recaller = _radar_recaller()
    rule = replace(
        create_rule(condition="", action=None, groups=[]),
        title="Refund Policy",
        content=RuleContent(
            condition="",
            action=None,
            description="Refunds are allowed within 30 days.",
        ),
        signals=["I want my money back"],
    )

    assert recaller._list_rule_contents(rule) == [
        "# Refund Policy\n\nRefunds are allowed within 30 days.",
        "I want my money back",
    ]


async def test_that_the_recaller_reuses_the_cached_policy_frame() -> None:
    embedder = _FakeEmbedder()
    recaller = _radar_recaller(embedder)
    rules = list(_create_sample_rules().values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    await recaller.discover(context, rules)
    await recaller.discover(context, rules)

    assert len(embedder.embed_calls) == 2
    assert len(embedder.embed_calls[0]) == 3
    assert len(embedder.embed_calls[1]) == 1


async def test_that_the_recaller_uses_the_persistent_embedding_cache() -> None:
    embedder = _FakeEmbedder()
    embedding_cache = _FakeEmbeddingCache()
    recaller_1 = _radar_recaller(embedder, embedding_cache)
    recaller_2 = _radar_recaller(embedder, embedding_cache)
    rules = list(_create_sample_rules().values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    await recaller_1.discover(context, rules)
    await recaller_2.discover(context, rules)

    assert len(embedder.embed_calls) == 2


async def test_that_the_recaller_returns_nothing_for_an_empty_interaction() -> None:
    recaller = _radar_recaller()
    rules = _create_sample_rules()

    context = _context([])

    result = await recaller.discover(context, list(rules.values()))

    assert result.discovered_rules == []


async def test_that_the_recaller_includes_a_single_candidate_rule() -> None:
    recaller = _radar_recaller()
    rule = create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        groups=[],
    )
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    result = await recaller.discover(context, [rule])

    assert result.discovered_rules[0].rule.id == rule.id
    assert result.discovered_rules[0].is_relevant


async def test_that_retrain_reports_progress_and_warms_recall() -> None:
    recaller = _radar_recaller()
    rules = list(_create_sample_rules().values())
    agent = create_agent()

    seen: list[float] = []

    async def on_progress(percentage: float) -> None:
        seen.append(percentage)

    report = ProgressReport(on_progress)
    await recaller.retrain(agent.id, rules, report)

    assert report.percentage == 100.0
    assert seen and seen[-1] == 100.0

    # Recall for this agent now serves off the trained frame.
    context = _context([(EventSource.CUSTOMER, "I'd like my money back")], agent=agent)
    result = await recaller.discover(context, rules)
    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}
    assert relevance_by_id[rules[0].id] or any(relevance_by_id.values())


async def test_that_retrain_uses_the_flat_recall_margin_as_the_threshold() -> None:
    recaller = _radar_recaller()
    rules = list(_create_sample_rules().values())
    agent = create_agent()

    await recaller.retrain(agent.id, rules)

    frame = recaller._frames_by_agent[agent.id]
    for rule in rules:
        policy = frame.by_rule[rule.id]
        assert math.isfinite(policy.threshold)
        assert policy.threshold == -RuleRecaller.DEFAULT_RECALL_MARGIN


async def test_that_recall_scores_are_returned_as_normalized_display_scores() -> None:
    recaller = _radar_recaller()
    rules = list(_create_sample_rules().values())
    context = _context([(EventSource.CUSTOMER, "I'd like my money back")])

    result = await recaller.discover(context, rules)

    for recalled in result.discovered_rules:
        assert 0.0 <= recalled.score <= 1.0
        if recalled.is_relevant:
            assert recalled.score > 0.5
        else:
            assert recalled.score <= 0.5


async def test_that_each_agent_gets_its_own_trained_frame() -> None:
    recaller = _radar_recaller()
    rules = list(_create_sample_rules().values())
    agent_a = create_agent()
    agent_b = create_agent()

    await recaller.retrain(agent_a.id, rules)

    # Agent A is trained; agent B is not — frames are strictly per-agent.
    assert agent_a.id in recaller._frames_by_agent
    assert agent_b.id not in recaller._frames_by_agent


async def test_that_a_pinned_signal_forces_recall() -> None:
    hours = create_rule(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        groups=[],
    )
    refund = create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        groups=[],
    )
    shipping = create_rule(
        condition="the customer asks where their package is",
        action="share the shipping status",
        groups=[],
    )
    context = _context([(EventSource.CUSTOMER, "I want my money back")])

    plain = await _radar_recaller(recall_margin=0.0).discover(context, [hours, refund, shipping])
    plain_hours = next(r for r in plain.discovered_rules if r.rule.id == hours.id)
    assert not plain_hours.is_relevant

    pinned_hours = replace(hours, signals=["[__pin__]I want my money back"])
    pinned = await _radar_recaller(pin_match_epsilon=0.5).discover(
        context, [pinned_hours, refund, shipping]
    )
    pinned_result = next(r for r in pinned.discovered_rules if r.rule.id == hours.id)
    assert pinned_result.is_relevant
    assert pinned_result.score > 0.5


async def test_that_pin_prefixed_signals_become_must_fire_exemplars() -> None:
    recaller = _radar_recaller(pin_match_epsilon=0.5)
    hours = replace(
        create_rule(
            condition="the customer asks about opening hours",
            action="tell them the store hours",
            groups=[],
        ),
        signals=["[__pin__]I want my money back"],
    )
    refund = create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        groups=[],
    )

    agent = create_agent()
    await recaller.retrain(agent.id, [hours, refund])

    assert len(recaller._frames_by_agent[agent.id].by_rule[hours.id].pin_exemplars) == 1


# --- Readmission offsets (anti-flapping for evicted session rules) --------
#
# The recaller deliberately max-pools over every historical customer message
# ("stickiness"). An evicted session rule must NOT be resurrected by the very
# message that originally triggered it: under a readmission offset, only customer
# messages strictly AFTER that offset may fire it — and the cumulative full-history
# query (which contains the original trigger) is excluded for it entirely.


async def test_that_recall_ignores_customer_messages_at_or_before_a_rule_readmission_offset() -> (
    None
):
    recaller = _radar_recaller()
    rules = _create_sample_rules()
    available = list(rules.values())
    # The refund trigger is at offset 0; the conversation has since moved on.
    context = _context(
        [
            (EventSource.CUSTOMER, "I want my money back"),
            (EventSource.AI_AGENT, "Let me help."),
            (EventSource.CUSTOMER, "what are your opening hours?"),
        ]
    )

    context.state.evicted_session_rules = {rules["refund"].id: 2}

    result = await recaller.discover(context, available)

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}

    # Without the offset the sticky max-pool keeps refund relevant (see
    # test_that_the_recaller_stays_sticky_across_user_turns); the floor kills that.
    assert not relevance_by_id[rules["refund"].id]
    # Other rules are unaffected by someone else's floor.
    assert relevance_by_id[rules["hours"].id]


async def test_that_recall_readmits_a_rule_when_a_new_customer_message_after_eviction_matches() -> (
    None
):
    recaller = _radar_recaller()
    rules = _create_sample_rules()
    available = list(rules.values())
    # Evicted at offset 1; the customer then raises the refund topic afresh.
    context = _context(
        [
            (EventSource.CUSTOMER, "what are your opening hours?"),
            (EventSource.AI_AGENT, "We're open 9-6."),
            (EventSource.CUSTOMER, "actually, I want my money back"),
        ]
    )

    context.state.evicted_session_rules = {rules["refund"].id: 1}

    result = await recaller.discover(context, available)

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}

    assert relevance_by_id[rules["refund"].id]


async def test_that_recall_without_readmission_offsets_is_unchanged() -> None:
    recaller = _radar_recaller()
    rules = _create_sample_rules()
    available = list(rules.values())
    context = _context(
        [
            (EventSource.CUSTOMER, "I want my money back"),
            (EventSource.AI_AGENT, "Let me help."),
            (EventSource.CUSTOMER, "what are your opening hours?"),
        ]
    )

    baseline = await recaller.discover(context, available)
    context.state.evicted_session_rules = {}
    explicit_empty = await recaller.discover(context, available)

    baseline_by_id = {r.rule.id: r.is_relevant for r in baseline.discovered_rules}
    explicit_by_id = {r.rule.id: r.is_relevant for r in explicit_empty.discovered_rules}

    # Stickiness preserved for non-ledgered rules: the old refund trigger
    # still fires it.
    assert baseline_by_id[rules["refund"].id]
    assert baseline_by_id == explicit_by_id


async def test_that_single_rule_recall_respects_the_readmission_offset() -> None:
    # The single-candidate shortcut is unconditionally relevant today; under a
    # readmission floor it must first check that any customer message actually
    # arrived after the eviction.
    recaller = _radar_recaller()
    rule = create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        groups=[],
    )
    context = _context(
        [
            (EventSource.CUSTOMER, "I want my money back"),
            (EventSource.AI_AGENT, "Let me help."),
        ]
    )

    context.state.evicted_session_rules = {rule.id: 1}
    floored = await recaller.discover(context, [rule])
    assert not floored.discovered_rules or not floored.discovered_rules[0].is_relevant

    fresh_context = _context(
        [
            (EventSource.CUSTOMER, "I want my money back"),
            (EventSource.AI_AGENT, "Let me help."),
            (EventSource.CUSTOMER, "yes, please refund me"),
        ]
    )
    fresh_context.state.evicted_session_rules = {rule.id: 1}
    readmitted = await recaller.discover(fresh_context, [rule])
    assert readmitted.discovered_rules
    assert readmitted.discovered_rules[0].is_relevant


def test_that_the_rule_recaller_implements_the_rule_discoverer_port() -> None:
    from parlant.core.engines.compass.matching.rule_discovery import RuleDiscoverer

    assert isinstance(_radar_recaller(), RuleDiscoverer)
