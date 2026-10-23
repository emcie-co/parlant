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

from parlant.core.canned_responses import (
    CannedResponseStore,
    CannedResponseVectorStore,
)
from parlant.core.container import Container


async def test_that_a_relevance_hit_with_no_stored_response_keeps_the_other_scores_on_their_own_responses(
    container: Container,
) -> None:
    """A vector hit can outlive the response it points at — the vector index and
    the document collection are written separately, so a response deleted from
    one can still be matched through the other.

    Each surviving response must keep its own score. Pairing the responses that
    were found against the full list of hits, position by position, hands every
    response after the missing one a score that belongs to its neighbour.
    """
    store = cast(CannedResponseVectorStore, container[CannedResponseStore])

    responses = [
        await store.create_canned_response(value="Your order has shipped and is on its way."),
        await store.create_canned_response(value="Our opening hours are nine to five."),
        await store.create_canned_response(value="I can help you reset your password."),
    ]

    query = "where is my package"

    before = await store.filter_relevant_canned_responses(query, responses, max_count=3)

    scores_before = {result.canned_response.id: result.score for result in before}

    assert len(before) == 3
    # Distinct scores, or a score landing on the wrong response is invisible.
    assert len(set(scores_before.values())) == 3

    # Delete the closest hit's document while leaving its vector in place. The
    # caller still offers it as available, exactly as a stale index would.
    missing = max(before, key=lambda result: result.score).canned_response

    await store._canreps_collection.delete_one({"id": {"$eq": missing.id}})

    after = await store.filter_relevant_canned_responses(query, responses, max_count=3)

    assert [result.canned_response.id for result in after] == [
        result.canned_response.id for result in before if result.canned_response.id != missing.id
    ]

    for result in after:
        assert result.score == scores_before[result.canned_response.id]
