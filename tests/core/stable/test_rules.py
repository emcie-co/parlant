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

from datetime import datetime, timezone
from typing import Any, Optional, cast

import pytest

from parlant.core.agents import Effort
from parlant.core.common import Weight, generate_id
from parlant.core.container import Container

from parlant.core.rules import (
    Rule,
    RuleId,
    RuleStore,
    RuleVectorStore,
    compose_rule_query,
)
from tests.test_utilities import DeterministicEmbedder


def _rule(
    description: Optional[str] = None,
    condition: Optional[str] = None,
    title: Optional[str] = None,
) -> Rule:
    now = datetime.now(timezone.utc)

    return Rule(
        id=RuleId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        enabled=True,
        groups=[],
        metadata={},
        weight=Weight.MEDIUM,
        content=description,
        condition=condition,
        title=title,
    )


# ─────────────────────────── embedding format ───────────────────────────────
#
# The variants the embedding content must produce, treating None / empty /
# whitespace-only as absent.


async def test_that_a_full_rule_embeds_title_condition_and_description_blocks(
    container: Container,
) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(
            _rule(
                title="Greetings",
                condition="the customer greets",
                description="be warm and concise",
            )
        )
        == "# Greetings\n\nCondition: the customer greets\n\nbe warm and concise"
    )


async def test_that_condition_only_embeds_with_a_condition_label(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_rule(condition="the customer greets"))
        == "Condition: the customer greets"
    )


async def test_that_description_only_embeds_as_a_plain_block(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_rule(description="be warm and concise"))
        == "be warm and concise"
    )


async def test_that_condition_and_description_embed_as_separate_blocks(
    container: Container,
) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(
            _rule(condition="the customer greets", description="be warm and concise")
        )
        == "Condition: the customer greets\n\nbe warm and concise"
    )


async def test_that_title_and_description_embed_as_a_heading_and_a_block(
    container: Container,
) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_rule(title="Greetings", description="be warm and concise"))
        == "# Greetings\n\nbe warm and concise"
    )


async def test_that_whitespace_only_fields_are_treated_as_absent(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(
            _rule(title="  ", condition="the customer greets", description="  ")
        )
        == "Condition: the customer greets"
    )


async def test_that_a_rule_with_no_embeddable_content_raises(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])

    with pytest.raises(ValueError):
        store._rule_embedding_content(_rule(description="   "))


# ───────────────────────────── signals ──────────────────────────────────────


async def test_that_signals_are_listed_as_independent_embedding_contents(
    container: Container,
) -> None:
    store = cast(RuleVectorStore, container[RuleStore])

    rule = await store.create_rule(
        condition="the customer wants a refund",
        signals=["money back", "reimbursement"],
    )

    contents = store._list_rule_contents(rule)

    assert contents == [
        "Condition: the customer wants a refund",
        "money back",
        "reimbursement",
    ]


async def test_that_a_rule_can_be_created_with_signals(container: Container) -> None:
    store = container[RuleStore]

    rule = await store.create_rule(
        condition="the customer wants a refund",
        signals=["money back", "reimbursement"],
    )

    read = await store.read_rule(rule.id)

    assert set(read.signals) == {"money back", "reimbursement"}


async def test_that_a_rule_can_be_created_with_anti_signals(container: Container) -> None:
    store = container[RuleStore]

    rule = await store.create_rule(
        condition="the customer wants a refund",
        signals=["money back"],
        anti_signals=["store hours", "shipping status"],
    )

    read = await store.read_rule(rule.id)

    assert set(read.signals) == {"money back"}
    assert set(read.anti_signals) == {"store hours", "shipping status"}


async def test_that_signals_can_be_updated(container: Container) -> None:
    store = container[RuleStore]

    rule = await store.create_rule(
        condition="the customer wants a refund",
        signals=["money back"],
    )

    updated = await store.update_rule(rule.id, {"signals": ["reimbursement", "chargeback"]})

    assert set(updated.signals) == {"reimbursement", "chargeback"}


async def test_that_anti_signals_can_be_updated(container: Container) -> None:
    store = container[RuleStore]

    rule = await store.create_rule(
        condition="the customer wants a refund",
        anti_signals=["store hours"],
    )

    updated = await store.update_rule(
        rule.id, {"anti_signals": ["shipping status", "opening hours"]}
    )

    assert set(updated.anti_signals) == {"shipping status", "opening hours"}


async def test_that_rule_effort_can_be_created_and_updated(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    cast(Any, store._vector_collection)._embedder = DeterministicEmbedder(
        id="test/core-rule-effort", dimensions=3072
    )

    rule = await store.create_rule(
        condition="the customer requests a regulated action",
        effort=Effort.HIGH,
    )

    assert (await store.read_rule(rule.id)).effort == Effort.HIGH

    await store.update_rule(rule.id, {"effort": Effort.MAX})

    assert (await store.read_rule(rule.id)).effort == Effort.MAX


# ───────────────────────────── retrieval ────────────────────────────────────


async def test_that_list_rules_returns_all_without_embedding_search(
    container: Container,
) -> None:
    store = container[RuleStore]

    first = await store.create_rule(condition="the customer greets")
    second = await store.create_rule(condition="the customer leaves")

    listed_ids = {g.id for g in await store.list_rules()}

    assert {first.id, second.id} <= listed_ids


async def test_that_find_relevant_rules_matches_by_signal(container: Container) -> None:
    store = container[RuleStore]

    refund = await store.create_rule(
        condition="the customer wants a refund",
        signals=["I want my money back", "reimbursement please"],
    )
    weather = await store.create_rule(
        condition="the customer asks about the weather",
    )

    results = await store.find_relevant_rules(
        query="can I get my money back?",
        available_rules=[refund, weather],
        max_count=1,
    )

    assert [r.rule.id for r in results] == [refund.id]


# ─────────────────────────── rule query text ────────────────────────────
#
# `Rule.query` is the canonical "what does this rule talk about" text,
# used to find the glossary terms a rule depends on (at matching time and
# during indexing).


def test_that_a_rule_query_includes_title_condition_and_description() -> None:
    rule = _rule(
        title="PRS Escalation",
        condition="the customer reports PRS",
        description="PRS cases are handled by the medical desk.",
    )

    query = rule.query

    assert "PRS Escalation" in query
    assert "the customer reports PRS" in query
    assert "PRS cases are handled by the medical desk." in query


def test_that_compose_rule_query_skips_absent_parts() -> None:
    query = compose_rule_query(
        title=None,
        condition="the customer wants a refund",
        content="  ",
    )

    assert "the customer wants a refund" in query
    assert "None" not in query
    assert query == query.strip()


def test_that_compose_rule_query_matches_the_rule_property() -> None:
    rule = _rule(title="A Title", condition="a condition", description="a description")

    assert rule.query == compose_rule_query(
        title="A Title",
        condition="a condition",
        content="a description",
    )


# ─────────────────────────── flat rule surface ───────────────────────────────


async def test_that_a_rule_stores_flat_description_condition_and_title(
    container: Container,
) -> None:
    store = container[RuleStore]

    policy = await store.create_rule(
        content="Offer a refund when the customer is unsatisfied.",
        title="Refunds",
    )
    observation = await store.create_rule(
        condition="the customer is discussing billing",
        content="Billing is the active topic.",
    )

    stored_policy = await store.read_rule(policy.id)
    assert stored_policy.content == "Offer a refund when the customer is unsatisfied."
    assert stored_policy.title == "Refunds"
    assert stored_policy.condition is None

    stored_observation = await store.read_rule(observation.id)
    assert stored_observation.condition == "the customer is discussing billing"
    assert stored_observation.content == "Billing is the active topic."


async def test_that_a_rule_requires_a_description_or_a_condition(
    container: Container,
) -> None:
    store = container[RuleStore]

    with pytest.raises(ValueError):
        await store.create_rule(title="Only a title")


async def test_that_find_rule_matches_on_the_authored_trio(
    container: Container,
) -> None:
    store = container[RuleStore]

    refunds = await store.create_rule(
        content="Offer a refund when the customer is unsatisfied.",
        title="Refunds",
    )
    await store.create_rule(
        content="Offer a refund when the customer is unsatisfied.",
        title="Returns",
    )

    found = await store.find_rule(
        content="Offer a refund when the customer is unsatisfied.",
        title="Refunds",
    )
    assert found.id == refunds.id
