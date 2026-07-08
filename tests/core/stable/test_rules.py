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

from typing import cast

from lagom import Container

from parlant.core.rules import (
    RuleContent,
    RuleStore,
    RuleVectorStore,
)


def _content(
    condition: str = "",
    action: str | None = None,
    description: str | None = None,
) -> RuleContent:
    return RuleContent(condition=condition, action=action, description=description)


# ─────────────────────────── embedding format ───────────────────────────────
#
# The six variants the embedding content must produce, treating None / empty /
# whitespace-only as absent.


async def test_that_condition_and_action_embed_as_when_then(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_content("the customer greets", "greet them back"))
        == "When the customer greets, then greet them back"
    )


async def test_that_full_content_appends_a_description_block(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(
            _content("the customer greets", "greet them back", "be warm and concise")
        )
        == "When the customer greets, then greet them back\n\nDescription: be warm and concise"
    )


async def test_that_condition_only_embeds_with_a_condition_label(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_content("the customer greets"))
        == "Condition: the customer greets"
    )


async def test_that_condition_and_description_embed_with_labels(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_content("the customer greets", None, "be warm and concise"))
        == "Condition: the customer greets\n\nDescription: be warm and concise"
    )


async def test_that_action_only_embeds_with_an_action_label(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_content("", "greet them back")) == "Action: greet them back"
    )


async def test_that_action_and_description_embed_with_labels(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_content("", "greet them back", "be warm and concise"))
        == "Action: greet them back\n\nDescription: be warm and concise"
    )


async def test_that_whitespace_only_fields_are_treated_as_absent(container: Container) -> None:
    store = cast(RuleVectorStore, container[RuleStore])
    assert (
        store._rule_embedding_content(_content("the customer greets", "   ", "  "))
        == "Condition: the customer greets"
    )


# ───────────────────────────── signals ──────────────────────────────────────


async def test_that_signals_are_listed_as_independent_embedding_contents(
    container: Container,
) -> None:
    store = cast(RuleVectorStore, container[RuleStore])

    rule = await store.create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["money back", "reimbursement"],
    )

    contents = store._list_rule_contents(rule)

    assert contents == [
        "When the customer wants a refund, then start the refund flow",
        "money back",
        "reimbursement",
    ]


async def test_that_a_rule_can_be_created_with_signals(container: Container) -> None:
    store = container[RuleStore]

    rule = await store.create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["money back", "reimbursement"],
    )

    read = await store.read_rule(rule.id)

    assert set(read.signals) == {"money back", "reimbursement"}


async def test_that_a_rule_can_be_created_with_anti_signals(container: Container) -> None:
    store = container[RuleStore]

    rule = await store.create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
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
        action="start the refund flow",
        signals=["money back"],
    )

    updated = await store.update_rule(rule.id, {"signals": ["reimbursement", "chargeback"]})

    assert set(updated.signals) == {"reimbursement", "chargeback"}


async def test_that_anti_signals_can_be_updated(container: Container) -> None:
    store = container[RuleStore]

    rule = await store.create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        anti_signals=["store hours"],
    )

    updated = await store.update_rule(
        rule.id, {"anti_signals": ["shipping status", "opening hours"]}
    )

    assert set(updated.anti_signals) == {"shipping status", "opening hours"}


# ───────────────────────────── retrieval ────────────────────────────────────


async def test_that_list_rules_returns_all_without_embedding_search(
    container: Container,
) -> None:
    store = container[RuleStore]

    first = await store.create_rule(condition="the customer greets", action="greet back")
    second = await store.create_rule(condition="the customer leaves", action="say goodbye")

    listed_ids = {g.id for g in await store.list_rules()}

    assert {first.id, second.id} <= listed_ids


async def test_that_find_relevant_rules_matches_by_signal(container: Container) -> None:
    store = container[RuleStore]

    refund = await store.create_rule(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["I want my money back", "reimbursement please"],
    )
    weather = await store.create_rule(
        condition="the customer asks about the weather",
        action="give the forecast",
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
# during evaluation).


def test_that_a_rule_query_includes_title_condition_action_and_description() -> None:
    from datetime import datetime, timezone

    from parlant.core.common import Weight, generate_id
    from parlant.core.rules import Rule, RuleId

    rule = Rule(
        id=RuleId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        content=_content(
            condition="the customer reports PRS",
            action="escalate to an allergy specialist",
            description="PRS cases are handled by the medical desk.",
        ),
        enabled=True,
        groups=[],
        metadata={},
        weight=Weight.MEDIUM,
        title="PRS Escalation",
    )

    query = rule.query

    assert "PRS Escalation" in query
    assert "the customer reports PRS" in query
    assert "escalate to an allergy specialist" in query
    assert "PRS cases are handled by the medical desk." in query


def test_that_compose_rule_query_skips_absent_parts() -> None:
    from parlant.core.rules import compose_rule_query

    query = compose_rule_query(
        title=None,
        condition="the customer wants a refund",
        action=None,
        description="  ",
    )

    assert "the customer wants a refund" in query
    assert "None" not in query
    assert query == query.strip()


def test_that_compose_rule_query_matches_the_rule_property() -> None:
    from datetime import datetime, timezone

    from parlant.core.common import Weight, generate_id
    from parlant.core.rules import Rule, RuleId, compose_rule_query

    rule = Rule(
        id=RuleId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        modified_utc=datetime.now(timezone.utc),
        content=_content(condition="a condition", action="an action"),
        enabled=True,
        groups=[],
        metadata={},
        weight=Weight.MEDIUM,
        title="A Title",
    )

    assert rule.query == compose_rule_query(
        title="A Title",
        condition="a condition",
        action="an action",
        description=None,
    )
