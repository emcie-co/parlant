# Copyright 2024 Emcie Co Ltd.
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
from lagom import Container

from parlant.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionSchema,
    GuidelinePropositionShot,
    GuidelinePropositionsSchema,
)
from parlant.core.engines.alpha.message_event_generator import (
    MessageEventGenerator,
    MessageEventGeneratorShot,
    MessageEventSchema,
    Revision,
)
from parlant.core.engines.alpha.tool_caller import ToolCallInferenceSchema, ToolCallerInferenceShot
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.common import (
    EventId,
    GuidelineContent,
)
from parlant.core.sessions import Event
from parlant.core.shots import ShotCollection


async def test_that_appended_shot_is_displayed_in_guideline_proposer_prompt(
    container: Container,
) -> None:
    guideline_proposer = container[GuidelineProposer]
    shot_collection = container[ShotCollection[GuidelinePropositionShot]]

    guideline_content = GuidelineContent(
        condition="A user compliments the product",
        action="Thank him with in Portuguese",
    )

    new_shot = GuidelinePropositionShot(
        description="Test Shot Description",
        interaction_events=[
            Event(
                id=EventId("test_id"),
                source="customer",
                kind="message",
                creation_utc=datetime.now(timezone.utc),
                offset=0,
                correlation_id="",
                data={"message": "This is a very cool feature, man!"},
                deleted=False,
            )
        ],
        guidelines=[guideline_content],
        expected_result=GuidelinePropositionsSchema(
            checks=[
                GuidelinePropositionSchema(
                    guideline_number=1,
                    condition=guideline_content.condition,
                    condition_application_rationale="Rationale: The user said that the feature is cool",
                    condition_applies=True,
                    action=guideline_content.action,
                    guideline_previously_applied="fully",
                    applies_score=8,
                )
            ]
        ),
    )

    await shot_collection.append(new_shot)

    shots = await guideline_proposer.shots()

    assert new_shot in shots


async def test_that_appended_shot_is_displayed_in_tool_caller_shots(
    container: Container,
) -> None:
    tool_caller = container[ToolEventGenerator].tool_caller
    shot_collection = container[ShotCollection[ToolCallerInferenceShot]]

    new_shot = ToolCallerInferenceShot(
        description="Test Shot Description",
        context="Test shot - checking if appended shot is reflected by tool_caller.shots()",
        expected_result=ToolCallInferenceSchema(
            last_customer_message="Testing shot append",
            most_recent_customer_inquiry_or_need="Verifying tool caller logic",
            most_recent_customer_inquiry_or_need_was_already_resolved=False,
            name="test_tool",
            subtleties_to_be_aware_of="",
            tool_calls_for_candidate_tool=[],
        ),
    )

    await shot_collection.append(new_shot)

    shots = await tool_caller.shots()
    assert new_shot in shots


async def test_that_appended_shot_is_displayed_in_message_generator_shots(
    container: Container,
) -> None:
    message_generator = container[MessageEventGenerator]
    shot_collection = container[ShotCollection[MessageEventGeneratorShot]]

    new_shot = MessageEventGeneratorShot(
        description="Test Shot Description",
        expected_result=MessageEventSchema(
            last_message_of_customer="This is a very cool feature, man!",
            guidelines=[],
            insights=[],
            evaluation_for_each_instruction=[],
            revisions=[
                Revision(
                    revision_number=1,
                    content=("Muito Obrigado!"),
                    instructions_followed=[],
                    instructions_broken=[],
                    is_repeat_message=False,
                    followed_all_instructions=True,
                ),
            ],
        ),
    )

    await shot_collection.append(new_shot)

    shots = await message_generator.shots()
    assert new_shot in shots
