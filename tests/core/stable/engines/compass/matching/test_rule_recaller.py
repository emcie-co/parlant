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

import asyncio
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

import numpy as np
from parlant.core.container import Container
from pytest import fixture

from parlant.core.engines.compass.matching.rule_recaller import (
    RuleRecaller,
    _LogisticModel,
)
from parlant.core.engines.compass.context import ResponseState
from parlant.core.engines.engine_context import EngineContext
from parlant.core.rules import Rule
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.sessions import EventSource

from parlant.core.agents import Agent

from tests.core.stable.engines.compass.matching.utils import (
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
    def __init__(self, max_tokens: int = 8192) -> None:
        self.embed_calls: list[list[str]] = []
        self._max_tokens = max_tokens

    async def embed(self, texts: list[str], hints: Mapping[str, Any] = {}) -> EmbeddingResult:
        # Mimic the embedding API's hard input cap (e.g. OpenAI's 400 on
        # over-limit input), using the same word-count tokenization as
        # _FakeTokenizer.
        for text in texts:
            if len(text.split()) > self._max_tokens:
                raise RuntimeError(
                    f"embedding input of {len(text.split())} tokens exceeds "
                    f"the {self._max_tokens}-token model limit"
                )
        self.embed_calls.append(texts)
        return EmbeddingResult(vectors=[self._vector_for_text(text) for text in texts])

    @property
    def id(self) -> str:
        return "fake-radar-embedder"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

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


class _FakeEmbeddingCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, tuple[str, ...]], EmbeddingResult] = {}

    async def get(
        self, embedder: Embedder, texts: list[str], hints: Mapping[str, Any] = {}
    ) -> EmbeddingResult | None:
        return self._entries.get((embedder.id, tuple(texts)))

    async def set(
        self,
        embedder: Embedder,
        texts: list[str],
        vectors: list[list[float]],
        hints: Mapping[str, Any] = {},
    ) -> None:
        self._entries[(embedder.id, tuple(texts))] = EmbeddingResult(vectors=vectors)


class _FakeModelRouter:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def get_embedder(self, **kwargs: Any) -> Embedder:
        return self._embedder


def _radar_recaller(
    embedder: _FakeEmbedder | None = None,
    embedding_cache: _FakeEmbeddingCache | None = None,
    **kwargs: Any,
) -> RuleRecaller:
    return RuleRecaller(
        model_router=cast(Any, _FakeModelRouter(embedder or _FakeEmbedder())),
        tracer=create_engine_context(conversation=[]).tracer,
        embedding_cache=embedding_cache or _FakeEmbeddingCache(),  # type: ignore[arg-type]
        **kwargs,
    )


def _context(
    conversation: list[tuple[EventSource, str]],
    agent: Agent | None = None,
) -> EngineContext[Any, Any]:
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
    rule = create_rule(
        content="Refunds are allowed within 30 days.",
        title="Refund Policy",
        groups=[],
    )

    assert recaller._rule_embedding_content(rule) == (
        "# Refund Policy\n\nRefunds are allowed within 30 days."
    )


def test_that_the_recaller_embeds_policy_rule_text_and_signals() -> None:
    recaller = _radar_recaller()
    rule = replace(
        create_rule(
            content="Refunds are allowed within 30 days.",
            title="Refund Policy",
            groups=[],
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

    # The trained frame is reused: the rule signals (the 3-text corpus batch) are
    # embedded only once across both discovers. The per-turn query goes through the
    # embedder on each discover (queries no longer use the frame/persistent cache).
    signal_calls = [call for call in embedder.embed_calls if len(call) == 3]
    assert len(signal_calls) == 1


async def test_that_the_recaller_uses_the_persistent_embedding_cache_for_the_corpus() -> None:
    embedder = _FakeEmbedder()
    embedding_cache = _FakeEmbeddingCache()
    recaller_1 = _radar_recaller(embedder, embedding_cache)
    recaller_2 = _radar_recaller(embedder, embedding_cache)
    rules = list(_create_sample_rules().values())
    context = _context([(EventSource.CUSTOMER, "hi, I'd like to get my money back")])

    await recaller_1.discover(context, rules)
    await recaller_2.discover(context, rules)

    # The corpus (rule signals) is served from the shared persistent cache: embedded
    # once by recaller_1, then a cache hit for recaller_2. Per-turn queries don't use
    # the persistent cache, so they don't share through it.
    signal_calls = [call for call in embedder.embed_calls if len(call) == 3]
    assert len(signal_calls) == 1


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


async def test_that_retrain_offloads_the_classifier_fit_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-policy logistic fit is CPU-bound numpy; it must run off the event loop
    # (via asyncio.to_thread) so a cold-start retrain over many rules doesn't block
    # the loop and trip the event-loop health monitor.
    recaller = _radar_recaller()
    rules = list(_create_sample_rules().values())
    agent = create_agent()

    offloaded: list[str] = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

    await recaller.retrain(agent.id, rules)

    # The frame compute was dispatched to a worker thread, and the result is correct.
    assert "_fit_frame" in offloaded
    frame = recaller._frames_by_agent[agent.id]
    assert set(frame.by_rule.keys()) == {rule.id for rule in rules}


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


# --- Readmission offsets (anti-flapping for evicted discovered rules) --------
#
# The recaller deliberately max-pools over every historical customer message
# ("stickiness"). An evicted discovered rule must NOT be resurrected by the very
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

    context.state.evicted_discovered_rules = {rules["refund"].id: 2}

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

    context.state.evicted_discovered_rules = {rules["refund"].id: 1}

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
    context.state.evicted_discovered_rules = {}
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

    context.state.evicted_discovered_rules = {rule.id: 1}
    floored = await recaller.discover(context, [rule])
    assert not floored.discovered_rules or not floored.discovered_rules[0].is_relevant

    fresh_context = _context(
        [
            (EventSource.CUSTOMER, "I want my money back"),
            (EventSource.AI_AGENT, "Let me help."),
            (EventSource.CUSTOMER, "yes, please refund me"),
        ]
    )
    fresh_context.state.evicted_discovered_rules = {rule.id: 1}
    readmitted = await recaller.discover(fresh_context, [rule])
    assert readmitted.discovered_rules
    assert readmitted.discovered_rules[0].is_relevant


def test_that_the_rule_recaller_implements_the_rule_discoverer_port() -> None:
    from parlant.core.engines.compass.matching.rule_discovery import RuleDiscoverer

    assert isinstance(_radar_recaller(), RuleDiscoverer)


# --- Embedder token budget (chunking of oversized queries) --------------------
#
# Conversation-derived queries (the cumulative string and per-turn customer
# messages) are unbounded, while embedder models cap their input tokens. Queries
# over the budget must be chunked - the sticky max-pool absorbs extra rows - with
# each chunk inheriting its source query's offset semantics.

_NEUTRAL_FILLER = "the weather is nice today and the sky looks blue"


async def test_that_a_cumulative_query_exceeding_the_embedder_token_budget_is_chunked_and_recall_still_succeeds() -> (
    None
):
    embedder = _FakeEmbedder(max_tokens=32)
    recaller = _radar_recaller(embedder)
    rules = _create_sample_rules()
    available = list(rules.values())

    context = _context([(EventSource.CUSTOMER, _NEUTRAL_FILLER) for _ in range(6)])
    # The trigger lives only in the session summary, which is rendered solely in
    # the cumulative query - so recalling refund requires the oversized cumulative
    # query to survive, via its chunks.
    context.state.session_summary = "the customer wants their money back"

    result = await recaller.discover(context, available)

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}
    assert relevance_by_id[rules["refund"].id]
    assert all(len(text.split()) <= 32 for call in embedder.embed_calls for text in call)


async def test_that_chunked_per_turn_queries_preserve_their_event_offsets_for_readmission_floors() -> (
    None
):
    embedder = _FakeEmbedder(max_tokens=32)
    recaller = _radar_recaller(embedder)
    rules = _create_sample_rules()
    available = list(rules.values())

    # The readmission trigger arrives inside an oversized customer message
    # (after the floor); its chunks must carry the message's offset, or the
    # floor would wrongly exclude them.
    oversized_message = " ".join([_NEUTRAL_FILLER] * 4 + ["so anyway, I want my money back"])
    context = _context(
        [
            (EventSource.CUSTOMER, "what else can you tell me?"),
            (EventSource.AI_AGENT, "Let me check."),
            (EventSource.CUSTOMER, oversized_message),
        ]
    )
    context.state.evicted_discovered_rules = {rules["refund"].id: 1}

    result = await recaller.discover(context, available)

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}
    assert relevance_by_id[rules["refund"].id]


async def test_that_chunks_of_the_cumulative_query_remain_ineligible_for_readmission() -> None:
    embedder = _FakeEmbedder(max_tokens=32)
    recaller = _radar_recaller(embedder)
    rules = _create_sample_rules()
    available = list(rules.values())

    # The old trigger sits in an oversized message at offset 0, before the
    # floor; neither its own chunks (offset 0) nor the cumulative query's
    # chunks (never eligible) may resurrect the evicted rule.
    old_oversized_trigger = " ".join(
        ["I want my money back please, and also:"] + [_NEUTRAL_FILLER] * 4
    )
    context = _context(
        [
            (EventSource.CUSTOMER, old_oversized_trigger),
            (EventSource.AI_AGENT, "Handled."),
            (EventSource.CUSTOMER, "what are your opening hours?"),
        ]
    )
    context.state.evicted_discovered_rules = {rules["refund"].id: 2}

    result = await recaller.discover(context, available)

    relevance_by_id = {r.rule.id: r.is_relevant for r in result.discovered_rules}
    assert not relevance_by_id[rules["refund"].id]
    assert relevance_by_id[rules["hours"].id]


async def test_that_in_budget_queries_are_embedded_verbatim() -> None:
    embedder = _FakeEmbedder()
    recaller = _radar_recaller(embedder)
    rules = list(_create_sample_rules().values())
    conversation = [
        (EventSource.CUSTOMER, "hi, I'd like to get my money back"),
        (EventSource.AI_AGENT, "Sure, let me help."),
        (EventSource.CUSTOMER, "thanks, how long does it take?"),
    ]
    context = _context(conversation)

    await recaller.discover(context, rules)

    # Chunk texts are embedding-cache keys: an in-budget cumulative query must be
    # embedded byte-identically, never re-split. The canonical query form tags each
    # line with the source's value (see matching.common.render_message_query).
    embedded_texts = {text for call in embedder.embed_calls for text in call}
    expected_cumulative = "\n".join(
        f"{source.value}: {message}" for source, message in conversation
    )
    assert expected_cumulative in embedded_texts
