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

from lagom import Container

from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.engines.sigma.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.sessions import EventSource

from tests.core.stable.engines.sigma.guideline_matching.utils import (
    create_engine_context,
    create_guideline,
)


def test_that_a_guideline_recaller_can_be_created() -> None:
    assert GuidelineRecaller() is not None


async def test_that_the_recaller_recalls_a_relevant_guideline() -> None:
    guideline = create_guideline(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )

    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")],
    )

    result = await GuidelineRecaller().recall(context, [guideline])

    assert len(result.recalled_guidelines) == 1
    assert result.recalled_guidelines[0].guideline == guideline
    assert result.recalled_guidelines[0].relevant


# ─────────── embedder-backed retrieval over the real GuidelineStore ──────────
#
# These exercise find_relevant_guidelines() end-to-end: three guidelines are
# created in the actual (container-provided) vector store so their content and
# signals get embedded, then sample conversations are matched against them with
# max_count=1, expecting exactly the right one of three back.


async def _create_sample_guidelines(store: GuidelineStore) -> dict[str, Guideline]:
    refund = await store.create_guideline(
        condition="the customer wants a refund",
        action="start the refund flow",
        signals=["I want my money back", "this is broken, give me a refund"],
    )
    hours = await store.create_guideline(
        condition="the customer asks about opening hours",
        action="tell them the store hours",
        signals=["when do you open", "are you open on sunday"],
    )
    shipping = await store.create_guideline(
        condition="the customer asks where their order is",
        action="share the shipping status",
        signals=["where is my package", "track my delivery"],
    )

    return {"refund": refund, "hours": hours, "shipping": shipping}


async def test_that_find_relevant_guidelines_returns_the_single_right_guideline_per_conversation(
    container: Container,
) -> None:
    store = container[GuidelineStore]
    guidelines = await _create_sample_guidelines(store)
    available = list(guidelines.values())

    cases: list[tuple[str, str]] = [
        ("hi, I'd like to get my money back for this order", "refund"),
        ("what time do you open tomorrow?", "hours"),
        ("my package still hasn't arrived, where is it?", "shipping"),
    ]

    for last_customer_message, expected_key in cases:
        results = await store.find_relevant_guidelines(
            query=last_customer_message,
            available_guidelines=available,
            max_count=1,
        )

        assert len(results) == 1
        assert results[0].guideline.id == guidelines[expected_key].id, (
            f"query {last_customer_message!r} expected {expected_key}"
        )


async def test_that_find_relevant_guidelines_matches_via_a_signal_over_the_main_content(
    container: Container,
) -> None:
    store = container[GuidelineStore]
    guidelines = await _create_sample_guidelines(store)

    # Phrased to match the refund guideline's *signal* ("I want my money back")
    # rather than its condition/action wording.
    results = await store.find_relevant_guidelines(
        query="I want my money back please",
        available_guidelines=list(guidelines.values()),
        max_count=1,
    )

    assert len(results) == 1
    assert results[0].guideline.id == guidelines["refund"].id
