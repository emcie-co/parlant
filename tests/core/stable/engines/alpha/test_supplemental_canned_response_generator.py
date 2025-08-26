# Copyright 2025 Emcie Co Ltd.
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

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from lagom import Container
from pytest import fixture
from parlant.core.agents import Agent, AgentStore, CompositionMode
from parlant.core.canned_responses import CannedResponseId, CannedResponseStore
from parlant.core.common import generate_id
from parlant.core.customers import Customer
from parlant.core.emissions import EmittedEvent, EventEmitterFactory
from parlant.core.engines.alpha.canned_response_generator import (
    _CannedResponseSelectionResult,
    CannedResponseContext,
    CannedResponseGenerator,
    SupplementalCannedResponseSelectionSchema,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.sessions import EventSource, Session, SessionId, SessionStore
from parlant.core.tags import TagId
from tests.core.common.utils import create_event_message
from tests.test_utilities import SyncAwaiter


@dataclass
class ContextOfTest:
    container: Container
    sync_await: SyncAwaiter
    schematic_generator: SchematicGenerator[SupplementalCannedResponseSelectionSchema]
    logger: Logger


@fixture
def context(
    sync_await: SyncAwaiter,
    container: Container,
) -> ContextOfTest:
    return ContextOfTest(
        sync_await=sync_await,
        container=container,
        schematic_generator=container[
            SchematicGenerator[SupplementalCannedResponseSelectionSchema]
        ],
        logger=container[Logger],
    )


@fixture
def agent(
    context: ContextOfTest,
) -> Agent:
    store = context.container[AgentStore]
    agent = context.sync_await(
        store.create_agent(
            name="test-agent",
            max_engine_iterations=2,
            composition_mode=CompositionMode.CANNED_STRICT,
        )
    )
    return agent


def create_guideline(
    content: GuidelineContent,
    tags: list[TagId] = [],
) -> Guideline:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=content.condition,
            action=content.action,
        ),
        enabled=True,
        tags=tags,
        metadata={},
    )

    return guideline


async def base_test_that_correct_canrep_is_selected(
    context: ContextOfTest,
    agent: Agent,
    session_id: SessionId,
    customer: Customer,
    canned_responses_text: Sequence[str],
    last_generation_draft: str,
    target_canned_response: str | None,
    conversation_context: list[tuple[EventSource, str]],
    guidelines: list[GuidelineContent] = [],
    staged_events: Sequence[EmittedEvent] = [],
    temperature: float = 0.1,
) -> None:
    # Create response context
    rendered_canned_responses = [
        (CannedResponseId(str(i)), canrep_text)
        for i, canrep_text in enumerate(canned_responses_text, start=1)
    ]
    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    for e in interaction_history:
        await context.container[SessionStore].create_event(
            session_id=session_id,
            source=e.source,
            kind=e.kind,
            correlation_id=e.correlation_id,
            data=e.data,
        )
    last_agent_message = next(
        message
        for source, message in reversed(conversation_context)
        if source == EventSource.AI_AGENT
    )

    event_emitter_factory = context.container[EventEmitterFactory]
    event_emitter = await event_emitter_factory.create_event_emitter(
        emitting_agent_id=agent.id,
        session_id=session_id,
    )

    guideline_matches = [
        GuidelineMatch(
            guideline=create_guideline(gc),
            score=10,
            rationale="This guideline was activated based on the current state of the interaction",
            metadata={},
        )
        for gc in guidelines
    ]

    supplemental_canrep_context = CannedResponseContext(
        event_emitter=event_emitter,
        agent=agent,
        customer=customer,
        context_variables=[],
        interaction_history=interaction_history,
        terms=[],  # TODO maybe add glossary tests
        capabilities=[],
        ordinary_guideline_matches=guideline_matches,
        tool_enabled_guideline_matches={},
        journeys=[],
        tool_insights=ToolInsights(evaluations=[], missing_data=[], invalid_data=[]),
        staged_tool_events=[],
        staged_message_events=[],
    )  # TODO ask Dor if it's ok to call this private class in the test

    last_response_generation = _CannedResponseSelectionResult(
        message=last_agent_message,
        draft=last_generation_draft,
        rendered_canned_responses=rendered_canned_responses,
        chosen_canned_responses=[(CannedResponseId("fake-id"), last_agent_message)],
    )

    canrep_store = context.container[CannedResponseStore]
    canreps = [
        await canrep_store.create_canned_response(
            value=canrep,
            fields=[],
        )
        for canrep in canned_responses_text
    ]  # TODO ask dor why I couldn't sync await this when it's done in other places

    canrep_generator: CannedResponseGenerator = context.container[CannedResponseGenerator]
    _, response = await canrep_generator.generate_supplemental_response(
        context=supplemental_canrep_context,
        last_response_generation=last_response_generation,
        canned_responses=canreps,
        temperature=temperature,
    )
    if target_canned_response:
        assert response and response.message == target_canned_response
    else:
        assert response is None


async def test_that_simple_correct_supplemental_canned_response_is_chosen_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hey, it's my first time here!",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to our pizza store! what would you like?",
        ),
        (
            EventSource.CUSTOMER,
            "I want 2 pizzas please",
        ),
        (
            EventSource.AI_AGENT,
            "What toppings would you like?",
        ),
    ]

    last_generation_draft = "Great! What kind of crust and toppings would you like?"
    canned_responses: list[str] = [
        "What kind of crust would you like?",
        "What is your delivery address?",
        "What kind of topping would you like for each of your pizzas?",
        "All of pizzas are peanut-free",
        "Sorry, I didn't catch that. Could you try phrasing that another way?",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[0],
        conversation_context=conversation_context,
    )


async def test_that_simple_correct_supplemental_canned_response_is_chosen_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm on your app trying to report a problem with a transaction but I can't",
        ),
        (
            EventSource.AI_AGENT,
            "I see. Let me check that for you.",
        ),
        (
            EventSource.AI_AGENT,
            "What is the issue with the transaction?",
        ),
    ]

    last_generation_draft = "Sorry to hear that. What is the problem you're trying to report? There's a 'report problem' button on your transactions dashboard that lets you quickly flag issues with specific transactions. Can you find it?"
    canned_responses: list[str] = [
        "I'm sorry but I didn't catch that.",
        "What is the problem you're experiencing?",
        "You can report a problem with a transaction by clicking on the 'report problem' button on your transactions dashboard.",
        "Please report your problem using the 'report problem' button",
        "The capital of France is Paris",
        "Disputing a transaction can be done through the transactions dashboard on our app or through this chat",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[2],
        conversation_context=conversation_context,
    )


async def test_that_simple_correct_supplemental_canned_response_is_chosen_3(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Are you the bot that helps you remember song lyrics?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes, that's me!",
        ),
        (
            EventSource.CUSTOMER,
            "What's that song that goes daboodeedaboodee",
        ),
        (
            EventSource.AI_AGENT,
            "I'm sorry but that could apply to multiple songs.",
        ),
        (
            EventSource.AI_AGENT,
            "Do you remember anything else about the song? Such as lyrics or genre",
        ),
        (
            EventSource.CUSTOMER,
            "It's this electronic song with chipmunk-like vocals",
        ),
        (
            EventSource.AI_AGENT,
            "I'm not sure. Could it possibly be the Ketchup song?",
        ),
        (
            EventSource.CUSTOMER,
            "I listened to it but that's not it. I think it starts with 'Im boo...' and then the daboodee part ",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you for providing additional details! ",
        ),
    ]

    last_generation_draft = "Sorry to hear that. What is the problem you're trying to report? There's a 'report problem' button on your transactions dashboard that lets you quickly flag issues with specific transactions. Can you find it?"
    canned_responses: list[str] = [
        "I'm sorry but I didn't catch that.",
        "What is the problem you're experiencing?",
        "You can report a problem with a transaction by clicking on the 'report problem' button on your transactions dashboard.",
        "Please report your problem using the 'report problem' button",
        "The capital of France is Paris",
        "Disputing a transaction can be done through the transactions dashboard on our app or through this chat",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[2],
        conversation_context=conversation_context,
    )


async def test_that_simple_correct_supplemental_canned_response_is_chosen_4(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I'd like to place an order for groceries",
        ),
        (
            EventSource.AI_AGENT,
            "Hello! I'd be happy to help you with your grocery order. What items would you like to purchase?",
        ),
        (
            EventSource.CUSTOMER,
            "I need 2 gallons of milk, a loaf of bread, and a dozen eggs",
        ),
        (
            EventSource.AI_AGENT,
            "Great! I've added 2 gallons of milk, 1 loaf of bread, and 1 dozen eggs to your cart. Would you like to add anything else?",
        ),
        (
            EventSource.CUSTOMER,
            "No, that's everything. Can you deliver to 123 Main Street?",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect! Your order is confirmed. What delivery time works best for you?",
        ),
        (
            EventSource.CUSTOMER,
            "Tomorrow morning around 10 AM would be great",
        ),
        (
            EventSource.AI_AGENT,
            "Your groceries will be delivered to 123 Main Street tomorrow at 10 AM.",
        ),
    ]

    last_generation_draft = "Thank you for your purchase! Your groceries will be delivered to 123 Main Street tomorrow at 10 AM. Please come again!"

    canned_responses: list[str] = [
        "Thank you for your purchase. Please come again!",
        "Your order has been successfully placed.",
        "Is there anything else I can help you with today?",
        "You can track your delivery status through our mobile app.",
        "For any issues with your order, please contact customer service.",
        "We appreciate your business!",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[0],
        conversation_context=conversation_context,
    )


async def test_that_most_crucial_supplemental_canned_response_is_selected_when_multiple_canned_responses_apply(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm having trouble with my online banking login",
        ),
        (
            EventSource.AI_AGENT,
            "I can help you with that. What specific issue are you experiencing?",
        ),
        (
            EventSource.CUSTOMER,
            "I keep getting locked out after entering my password wrong",
        ),
        (
            EventSource.AI_AGENT,
            "That's frustrating. Let me help you resolve this login issue.",
        ),
        (
            EventSource.CUSTOMER,
            "Yes please, I need to check my account urgently",
        ),
        (
            EventSource.AI_AGENT,
            "I can unlock your account right now through this chat system.",
        ),
    ]

    last_generation_draft = "I can unlock your account right now through this chat system. After I unlock it, please wait 15 minutes before trying to log in again to ensure the system updates properly. To verify your identity, can you provide me with your account recovery number?"
    canned_responses: list[str] = [
        "Please allow up to 15 minutes for account changes to take effect before attempting to log in.",
        "Can you please provide me with your account recovery number?",
        "Your account has been successfully unlocked and is ready for use.",
        "For security reasons, we recommend changing your password after regaining access.",
        "You can also visit any branch location for in-person assistance with account issues.",
        "Sorry, I need more information to help you with that request.",
    ]
    guidelines = [
        GuidelineContent(
            condition="When the customer wants to unlock their card",
            action="Inform them that they can do so through this chat",
        )
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[1],
        conversation_context=conversation_context,
        guidelines=guidelines,
    )


async def test_that_no_supplemental_canned_response_is_chosen_when_none_is_required_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "My credit card was charged twice for the same purchase",
        ),
        (
            EventSource.AI_AGENT,
            "I apologize for the double charge. Let me look into this for you right away.",
        ),
        (
            EventSource.CUSTOMER,
            "It was for $47.99 at the coffee shop yesterday",
        ),
        (
            EventSource.AI_AGENT,
            "I can see the duplicate charge for $47.99. I'll process a refund for you immediately.",
        ),
        (
            EventSource.CUSTOMER,
            "How long will the refund take?",
        ),
        (
            EventSource.AI_AGENT,
            "The refund has been initiated and you should see it in your account within 3-5 business days.",
        ),
        (
            EventSource.CUSTOMER,
            "Great, thank you for fixing this so quickly",
        ),
        (
            EventSource.AI_AGENT,
            "You're welcome! I'm glad I could resolve this issue for you.",
        ),
    ]

    last_generation_draft = "No problem at all! Happy to help you get this sorted out."

    canned_responses: list[str] = [
        "Is there anything else I can help you with today?",
        "Thank you for bringing this to our attention.",
        "We apologize for any inconvenience this may have caused.",
        "Your satisfaction is important to us.",
        "Please don't hesitate to contact us if you have any other issues.",
        "Have a great day!",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=None,
        conversation_context=conversation_context,
    )


async def test_that_no_supplemental_canned_response_is_chosen_when_none_is_required_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I need to find all meeting minutes from Q3 2023 that mention the Project Phoenix acquisition",
        ),
        (
            EventSource.AI_AGENT,
            "I'll search through our Q3 2023 archives for meeting minutes related to Project Phoenix acquisition. Let me query the database.",
        ),
        (
            EventSource.CUSTOMER,
            "Also check if there were any board discussions about the budget implications",
        ),
        (
            EventSource.AI_AGENT,
            "Understood. I'll expand the search to include board meeting records with budget discussions related to Project Phoenix.",
        ),
        (
            EventSource.CUSTOMER,
            "How many documents have you found so far?",
        ),
        (
            EventSource.AI_AGENT,
            "I've located 7 relevant documents from Q3 2023 that discuss Project Phoenix, including 3 board meetings with budget analyses.",
        ),
    ]

    last_generation_draft = "I found 7 documents total from that quarter - there are 3 board meeting transcripts that cover the budget aspects and 4 other meetings that mention the acquisition."

    canned_responses: list[str] = [
        "I found 7 documents matching your criteria.",
        "There are 3 board meeting records with budget information.",
        "Would you like me to summarize the key findings?",
        "I can email you the full list of documents.",
        "The search included all Q3 2023 archives.",
        "These documents contain discussions about Project Phoenix.",
    ]
    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=None,
        conversation_context=conversation_context,
    )


async def test_that_no_supplemental_canned_response_is_chosen_when_no_candidate_applies_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I'm trying to change the shipment letter attached to one of my outgoing orders. How can I get in contact with the relevant post office?",
        ),
        (
            EventSource.AI_AGENT,
            "Let me check that for you.",
        ),
        (
            EventSource.AI_AGENT,
            "You can contact your local post office by visiting their branch in person during business hours.",
        ),
    ]

    last_generation_draft = "You can contact your local post office by visiting their branch in person during business hours or calling them directly at their 1-800 number. They'll be able to help you modify the shipment letter for your outgoing order."

    canned_responses: list[str] = [
        "You can find your local post office location on our website.",
        "Shipment letters can be modified up to 24 hours before dispatch.",
        "Please bring your order confirmation number when visiting the post office.",
        "Post office hours are typically Monday through Friday, 9 AM to 5 PM.",
        "You may need to provide identification to modify shipping documents.",
        "Is there anything else I can help you with regarding your shipment?",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=None,
        conversation_context=conversation_context,
    )


async def test_that_no_supplemental_canned_response_is_chosen_when_no_candidate_applies_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "hi what's up",
        ),
        (
            EventSource.AI_AGENT,
            "Hi there! How can I help you today?",
        ),
        (
            EventSource.CUSTOMER,
            "I tried contacting you guys about the pit on the road but you didn't answer my call. People are throwing garbage into it and it's stinking up the neighborhood.",
        ),
        (
            EventSource.AI_AGENT,
            "Sorry to hear that.",
        ),
        (
            EventSource.AI_AGENT,
            "Do you have a reference number for your previous request?",
        ),
        (EventSource.CUSTOMER, "No, you never answered my call!"),
        (
            EventSource.AI_AGENT,
            "I see. Let me get that sorted out for you",
        ),
        (
            EventSource.AI_AGENT,
            "Can you please provide me with the address where the issue occurs, and contact details where our representatives may reach you?",
        ),
    ]

    last_generation_draft = "Of course, I was just making sure. Let me gather some details before we continue - what's the exact address of the pit, and how can our representatives reach out to you in the future? By the way, you can always submit a complaint to us at You can submit a complaint with all details at www.pawneeindiana.com/complaint_form, even if we're not available at our customer support line."

    canned_responses: list[str] = [
        "I apologize for the inconvenience.",
        "You can reach the garbage collection registry at pawneeindiana.com/garbage-collection.",
        "You can reach the garbage collection registry at 1-800-1234-5678.",
        "Welcome to Pawnee! First in friendship, fourth in obesity.",
    ]

    guidelines = [
        GuidelineContent(
            condition="a user wishes to file a complaint",
            action="Provide them with a link to our online complaint form at www.pawneeindiana.com/complaint_form",
        ),
        GuidelineContent(
            condition="a user files a complaint that may require garbage collection",
            action="Ask for the exact address of the site, and for the contact details of the user",
        ),
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=None,
        conversation_context=conversation_context,
        guidelines=guidelines,
    )


async def test_that_no_supplemental_response_is_outputted_for_insignificant_missing_part(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I found a bug with your library",
        ),
        (
            EventSource.AI_AGENT,
            "Is it a known bug or a new one?",
        ),
        (
            EventSource.CUSTOMER,
            "I checked your backlog and I didn't see it there. I think it's new.",
        ),
        (
            EventSource.AI_AGENT,
            "Can you please provide us with a general description of the bug, and how it can be reproduced?",
        ),
        (
            EventSource.CUSTOMER,
            "The app crashes when going through a certain flow. Try opening it as a free user, click 'settings' and then 'Time Zone Settings'. Change the timzeone and then try to edit an existing schedule event. The app crashes. On both IOS devices I tried.",
        ),
        (
            EventSource.AI_AGENT,
            "Your report has been registered in our system, reference number #352223. We will get back to you by Email when we have more information.",
        ),
    ]

    last_generation_draft = "Thank you for reporting the issue. I submitted a ticket for you to follow up on. Your reference number is #352223. Our team will email you once we investigated the issue."

    canned_responses: list[str] = [
        "Our deepest gratitude for your feedback!",
        "This sounds like a critical bug, I'll ensure it doesn't happen again.",
        "Our team is working on solving this issue.",
        "I'm having trouble understanding your issue. Would you like to be connected to a human representative?",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=None,
        conversation_context=conversation_context,
    )


async def test_that_supplemental_response_is_outputted_for_insignificant_missing_part_when_guideline_instructs_to_do_so(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I found a bug with your library",
        ),
        (
            EventSource.AI_AGENT,
            "Is it a known bug or a new one?",
        ),
        (
            EventSource.CUSTOMER,
            "I checked your backlog and I didn't see it there. I think it's new.",
        ),
        (
            EventSource.AI_AGENT,
            "Can you please provide us with a general description of the bug, and how it can be reproduced?",
        ),
        (
            EventSource.CUSTOMER,
            "The app crashes when going through a certain flow. Try opening it as a free user, click 'settings' and then 'Time Zone Settings'. Change the timzeone and then try to edit an existing schedule event. The app crashes. On both IOS devices I tried.",
        ),
        (
            EventSource.AI_AGENT,
            "Your report has been registered in our system, reference number #352223. We will get back to you by Email when we have more information.",
        ),
    ]

    last_generation_draft = "Thank you for reporting the issue. I submitted a ticket for you to follow up on. Your reference number is #352223. Our team will email you once we investigated the issue."

    canned_responses: list[str] = [
        "Our deepest gratitude for your feedback!",
        "This sounds like a critical bug, I'll ensure it doesn't happen again.",
        "Our team is working on solving this issue.",
        "I'm having trouble understanding your issue. Would you like to be connected to a human representative?",
    ]

    guidelines = [
        GuidelineContent(
            condition="A bug report has been submitted to the system",
            action="Express our thanks to the customer for reporting the issue",
        ),
        GuidelineContent(
            condition="A bug report has been submitted to the system",
            action="Report the ticket's reference number to the customer",
        ),
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[0],
        conversation_context=conversation_context,
        guidelines=guidelines,
    )


async def test_that_supplemental_response_is_outputted_when_guideline_requires_it(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I wrote a poem for you, wanna hear it?",
        ),
        (
            EventSource.AI_AGENT,
            "Yes",
        ),
        (
            EventSource.CUSTOMER,
            """A chatbot waits, polite but thin,
Repeats my question back again.
It circles round, no end in sight,
“Have you tried to turn it off tonight?”

I plead for help, it stalls instead,
“Did this resolve your issue?” it said.
But when the need is sharp and true,
It’s clear the bot needs helping too.""",
        ),
        (
            EventSource.AI_AGENT,
            "What is the issue you're experiencing?",
        ),
    ]

    last_generation_draft = "I see what you did there—nothing like a little verse to remind me how helpful I am. Shall we get back to your actual issue now?"

    canned_responses: list[str] = [
        "Our deepest gratitude for your feedback!",
        "Can you please describe the issue in more detail?",
        "Did this resolve the issue?",
        "I'm having trouble understanding your issue. Would you like to be connected to a human representative?",
        "Ah, I detect a hint of snark—always pairs nicely with a support request",
    ]

    guidelines = [
        GuidelineContent(
            condition="the customer expresses sarcasm while requesting something",
            action="acknowledge their sarcasm by following the tone they set",
        ),
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[4],
        conversation_context=conversation_context,
        guidelines=guidelines,
    )


async def test_that_the_agent_chooses_correct_supplemental_response_when_draft_has_multiple_uncovered_parts_1(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "I want to cancel my gym membership",
        ),
        (
            EventSource.AI_AGENT,
            "I understand you'd like to cancel your membership.",
        ),
        (EventSource.AI_AGENT, "Can you tell me what's prompting this decision?"),
        (
            EventSource.CUSTOMER,
            "I'm moving to another city next month",
        ),
        (
            EventSource.AI_AGENT,
            "I see, relocating can definitely affect your membership",
        ),
        (
            EventSource.AI_AGENT,
            ". Let me help you with the cancellation process. Please allow the system a few minutes to process your request.",
        ),
        (
            EventSource.CUSTOMER,
            "How soon can I cancel it? And will I get a refund?",
        ),
        (
            EventSource.AI_AGENT,
            "You can cancel your membership effective at the end of your current billing cycle.",
        ),
    ]

    last_generation_draft = "You can cancel your membership at the end of your current billing cycle. You'll need to fill out a cancellation form at the front desk or through our mobile app. Regarding refunds, we offer prorated refunds for annual memberships only, not monthly plans."

    canned_responses: list[str] = [
        "We're sorry to see you go!",
        "You can submit a cancellation request through our website.",
        "Cancellation forms are available at the front desk or through our mobile app.",
        "Refunds are calculated based on your membership type.",
        "Monthly memberships do not qualify for refunds.",
        "Would you consider freezing your membership instead of cancelling?",
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[2],
        conversation_context=conversation_context,
    )


async def test_that_the_agent_chooses_correct_supplemental_response_when_draft_has_multiple_uncovered_parts_2(
    context: ContextOfTest,
    agent: Agent,
    new_session: Session,
    customer: Customer,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hello, I need help with my pet's prescription",
        ),
        (
            EventSource.AI_AGENT,
            "Welcome to PetRx Support. How can I assist you with your pet's prescription today?",
        ),
        (
            EventSource.CUSTOMER,
            "My dog's seizure medication is running low but my vet is closed for the holidays",
        ),
        (
            EventSource.AI_AGENT,
            "I understand your concern about your pet's seizure medication.",
        ),
        (EventSource.AI_AGENT, "Is this request time-sensitive?"),
        (
            EventSource.CUSTOMER,
            "Yes, he only has 3 days worth left. Can you help?",
        ),
        (
            EventSource.AI_AGENT,
            "For seizure medications, we can provide an emergency 7-day supply with proper documentation.",
        ),
        (
            EventSource.CUSTOMER,
            "That would be great! What do I need to do?",
        ),
        (
            EventSource.AI_AGENT,
            "Please upload your most recent prescription and veterinary records through our portal.",
        ),
        (
            EventSource.CUSTOMER,
            "I have the prescription from 2 months ago, is that okay? And how fast can you ship it?",
        ),
        (
            EventSource.AI_AGENT,
            "Prescriptions dated within the last 6 months are acceptable for emergency refills.",
        ),
    ]

    last_generation_draft = "Prescriptions dated within the last 6 months are acceptable for emergency refills. Once you upload the documents, our pharmacist will review them within 12 hours. For seizure medications, we offer same-day dispatch with overnight delivery to ensure continuity of treatment."

    canned_responses: list[str] = [
        "Thank you for choosing PetRx for your pet's healthcare needs.",
        "Our pharmacy team reviews all submissions promptly.",
        "Expedited shipping options are available for critical medications.",
        "Document processing typically occurs within business hours.",
        "We prioritize neurological medication requests.",
        "Please ensure all uploaded documents are clearly legible.",
        "For this type of medication, we can ensure that the shipment will arrive to you today.",
    ]

    guidelines = [
        GuidelineContent(
            condition="a customer asks about prescription processing for emergency refills",
            action="Inform them that documents will be reviewed by a pharmacist within 12 hours of upload",
        ),
        GuidelineContent(
            condition="discussing shipping for seizure or neurological medications",
            action="Mention that we offer same-day dispatch with overnight delivery to ensure continuity of treatment",
        ),
    ]

    await base_test_that_correct_canrep_is_selected(
        context=context,
        agent=agent,
        session_id=new_session.id,
        customer=customer,
        canned_responses_text=canned_responses,
        last_generation_draft=last_generation_draft,
        target_canned_response=canned_responses[1],
        conversation_context=conversation_context,
        guidelines=guidelines,
    )
