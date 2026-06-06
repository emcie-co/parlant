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

from parlant.core.engines.compass.guideline_matching.guideline_distiller import GuidelineDistiller
from parlant.core.sessions import EventSource

from tests.core.stable.engines.compass.guideline_matching.utils import (
    create_engine_context,
    create_guideline,
)


@fixture
def distiller(container: Container) -> GuidelineDistiller:
    return container[GuidelineDistiller]


def test_that_a_guideline_distiller_can_be_created(distiller: GuidelineDistiller) -> None:
    assert distiller is not None


async def test_that_the_distiller_distills_a_relevant_guideline(
    distiller: GuidelineDistiller,
) -> None:
    guideline = create_guideline(
        condition="the customer asks about toppings",
        action="list the available toppings",
    )

    context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "what toppings do you have?")],
    )

    result = await distiller.distill(context, [guideline])

    assert len(result.distilled_guidelines) == 1
    assert result.distilled_guidelines[0].guideline == guideline
    assert result.distilled_guidelines[0].is_relevant
