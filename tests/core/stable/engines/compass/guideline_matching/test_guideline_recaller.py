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
from pytest import fixture

from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.engines.compass.guideline_matching.guideline_recaller import GuidelineRecaller
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import create_engine_context


@fixture
def recaller(container: Container) -> GuidelineRecaller:
    return container[GuidelineRecaller]


def test_that_a_guideline_recaller_can_be_created(recaller: GuidelineRecaller) -> None:
    assert recaller is not None


# ─────────── embedder-backed recall over the real GuidelineStore ─────────────
#
# The recaller is given the store and uses find_relevant_guidelines under the
# hood. Three guidelines are created in the actual (container-provided) vector
# store so their content and signals get embedded; sample conversations are
# then recalled against them with max_count=1, expecting exactly the right one
# of three back.


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


async def test_that_the_recaller_returns_the_single_right_guideline_per_conversation(
    container: Container,
    recaller: GuidelineRecaller,
) -> None:
    guidelines = await _create_sample_guidelines(container[GuidelineStore])
    available = list(guidelines.values())

    cases: list[tuple[str, str]] = [
        ("hi, I'd like to get my money back for this order", "refund"),
        ("what time do you open tomorrow?", "hours"),
        ("my package still hasn't arrived, where is it?", "shipping"),
    ]

    for last_customer_message, expected_key in cases:
        context = create_engine_context(
            conversation=[(EventSource.CUSTOMER, last_customer_message)],
        )

        result = await recaller.recall(context, available, max_count=1)

        assert len(result.recalled_guidelines) == 1
        assert result.recalled_guidelines[0].guideline.id == guidelines[expected_key].id, (
            f"message {last_customer_message!r} expected {expected_key}"
        )
        assert result.recalled_guidelines[0].is_relevant


async def test_that_the_recaller_matches_via_a_signal_over_the_main_content(
    container: Container,
    recaller: GuidelineRecaller,
) -> None:
    guidelines = await _create_sample_guidelines(container[GuidelineStore])

    # Phrased to match the refund guideline's *signal* ("I want my money back")
    # rather than its condition/action wording.
    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "I want my money back please")],
    )

    result = await recaller.recall(context, list(guidelines.values()), max_count=1)

    assert len(result.recalled_guidelines) == 1
    assert result.recalled_guidelines[0].guideline.id == guidelines["refund"].id


async def test_that_the_recaller_returns_nothing_for_an_empty_interaction(
    container: Container,
    recaller: GuidelineRecaller,
) -> None:
    guidelines = await _create_sample_guidelines(container[GuidelineStore])

    # No interaction at all -> no query to recall against.
    context = create_engine_context(conversation=[])

    result = await recaller.recall(context, list(guidelines.values()), max_count=1)

    assert result.recalled_guidelines == []
