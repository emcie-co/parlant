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

from parlant.core.guidelines import (
    GuidelineContent,
    GuidelineStore,
    GuidelineVectorStore,
)


def _content(
    condition: str = "",
    action: str | None = None,
    description: str | None = None,
) -> GuidelineContent:
    return GuidelineContent(condition=condition, action=action, description=description)


# ─────────────────────────── embedding format ───────────────────────────────
#
# The six variants the embedding content must produce, treating None / empty /
# whitespace-only as absent.


async def test_that_condition_and_action_embed_as_when_then(container: Container) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])
    assert (
        store._guideline_embedding_content(_content("the customer greets", "greet them back"))
        == "When the customer greets, then greet them back"
    )


async def test_that_full_content_appends_a_description_block(container: Container) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])
    assert (
        store._guideline_embedding_content(
            _content("the customer greets", "greet them back", "be warm and concise")
        )
        == "When the customer greets, then greet them back\n\nDescription: be warm and concise"
    )


async def test_that_condition_only_embeds_with_a_condition_label(container: Container) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])
    assert (
        store._guideline_embedding_content(_content("the customer greets"))
        == "Condition: the customer greets"
    )


async def test_that_condition_and_description_embed_with_labels(container: Container) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])
    assert (
        store._guideline_embedding_content(
            _content("the customer greets", None, "be warm and concise")
        )
        == "Condition: the customer greets\n\nDescription: be warm and concise"
    )


async def test_that_action_only_embeds_with_an_action_label(container: Container) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])
    assert (
        store._guideline_embedding_content(_content("", "greet them back"))
        == "Action: greet them back"
    )


async def test_that_action_and_description_embed_with_labels(container: Container) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])
    assert (
        store._guideline_embedding_content(_content("", "greet them back", "be warm and concise"))
        == "Action: greet them back\n\nDescription: be warm and concise"
    )


async def test_that_whitespace_only_fields_are_treated_as_absent(container: Container) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])
    assert (
        store._guideline_embedding_content(_content("the customer greets", "   ", "  "))
        == "Condition: the customer greets"
    )


# ───────────────────────────── signals ──────────────────────────────────────


async def test_that_signals_are_listed_as_independent_embedding_contents(
    container: Container,
) -> None:
    store = cast(GuidelineVectorStore, container[GuidelineStore])

    guideline = await store.create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["money back", "reimbursement"],
    )

    contents = store._list_guideline_contents(guideline)

    assert contents == [
        "When the customer wants a refund, then start the refund flow",
        "money back",
        "reimbursement",
    ]


async def test_that_a_guideline_can_be_created_with_signals(container: Container) -> None:
    store = container[GuidelineStore]

    guideline = await store.create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["money back", "reimbursement"],
    )

    read = await store.read_guideline(guideline.id)

    assert set(read.signals) == {"money back", "reimbursement"}


async def test_that_signals_can_be_updated(container: Container) -> None:
    store = container[GuidelineStore]

    guideline = await store.create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["money back"],
    )

    updated = await store.update_guideline(
        guideline.id, {"signals": ["reimbursement", "chargeback"]}
    )

    assert set(updated.signals) == {"reimbursement", "chargeback"}


# ───────────────────────────── retrieval ────────────────────────────────────


async def test_that_list_guidelines_returns_all_without_embedding_search(
    container: Container,
) -> None:
    store = container[GuidelineStore]

    first = await store.create_guideline(condition="the customer greets", action="greet back")
    second = await store.create_guideline(condition="the customer leaves", action="say goodbye")

    listed_ids = {g.id for g in await store.list_guidelines()}

    assert {first.id, second.id} <= listed_ids


async def test_that_find_relevant_guidelines_matches_by_signal(container: Container) -> None:
    store = container[GuidelineStore]

    refund = await store.create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["I want my money back", "reimbursement please"],
    )
    weather = await store.create_guideline(
        condition="the customer asks about the weather",
        action="give the forecast",
    )

    results = await store.find_relevant_guidelines(
        query="can I get my money back?",
        available_guidelines=[refund, weather],
        max_count=1,
    )

    assert [r.guideline.id for r in results] == [refund.id]
