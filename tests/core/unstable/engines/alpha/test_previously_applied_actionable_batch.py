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
from datetime import datetime, timezone

from pytest import fixture
from parlant.core.agents import Agent
from parlant.core.capabilities import Capability, CapabilityId
from parlant.core.customers import Customer
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
)
from parlant.core.sessions import EventSource, Session
from tests.core.stable.engines.alpha.test_previously_applied_actionable_batch import (
    ContextOfTest,
    base_test_that_correct_guidelines_are_matched,
)
from tests.test_utilities import SyncAwaiter
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        container,
        sync_await,
        guidelines=list(),
        logger=container[Logger],
        schematic_generator=container[
            SchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema]
        ],
    )


async def test_that_partially_fulfilled_action_with_missing_behavioral_part_is_matched_again(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    capabilities = [
        Capability(
            id=CapabilityId("cap_123"),
            creation_utc=datetime.now(timezone.utc),
            title="Reset Password",
            description="The ability to send the customer an email with a link to reset their password. The password can only be reset via this link",
            signals=["reset password", "password"],
            tags=[],
        )
    ]
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, can you reset my password?",
        ),
        (
            EventSource.AI_AGENT,
            "Sure, for that I will need your email please so I will send you the password. What's your email address?",
        ),
        (
            EventSource.CUSTOMER,
            "I forgot what I was going to say, can you continue from the same point?",
        ),
    ]

    guidelines: list[str] = ["reset_password"]

    await base_test_that_correct_guidelines_are_matched(
        context,
        agent,
        new_session.id,
        customer,
        conversation_context,
        guidelines_target_names=guidelines,
        guidelines_names=guidelines,
        capabilities=capabilities,
    )
