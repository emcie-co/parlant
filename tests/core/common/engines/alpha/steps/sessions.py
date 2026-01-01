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
import json
from pytest_bdd import given, parsers
from typing import cast

from parlant.core.agents import Agent, AgentId
from parlant.core.customers import Customer, CustomerStore
from parlant.core.sessions import EventKind, EventSource, Session, SessionId, SessionStore

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, "an empty session", target_fixture="session_id")
def given_an_empty_session(
    context: ContextOfTest,
    agent_id: AgentId,
) -> SessionId:
    session_store = context.container[SessionStore]
    customer_store = context.container[CustomerStore]

    utc_now = datetime.now(timezone.utc)

    customer = context.sync_await(customer_store.create_customer("test_customer"))
    session = context.sync_await(
        session_store.create_session(
            creation_utc=utc_now,
            customer_id=customer.id,
            agent_id=agent_id,
        )
    )
    return session.id


@step(given, parsers.parse('an empty session with "{customer_name}"'), target_fixture="session_id")
def given_an_empty_session_with_customer(
    context: ContextOfTest,
    agent_id: AgentId,
    customer_name: str,
) -> SessionId:
    session_store = context.container[SessionStore]
    customer_store = context.container[CustomerStore]

    utc_now = datetime.now(timezone.utc)

    customer = next(
        (
            customer
            for customer in context.sync_await(customer_store.list_customers())
            if customer.name == customer_name
        ),
        context.sync_await(customer_store.create_customer(customer_name)),
    )

    session = context.sync_await(
        session_store.create_session(
            creation_utc=utc_now,
            customer_id=customer.id,
            agent_id=agent_id,
        )
    )
    return session.id


@step(given, "a session with a single customer message", target_fixture="session_id")
def given_a_session_with_a_single_customer_message(
    context: ContextOfTest,
    new_session: Session,
    customer: Customer,
) -> SessionId:
    store = context.container[SessionStore]

    context.sync_await(
        store.create_event(
            session_id=new_session.id,
            source=EventSource.CUSTOMER,
            kind=EventKind.MESSAGE,
            trace_id="<main>",
            data={
                "message": "Hey there",
                "participant": {
                    "id": customer.id,
                    "display_name": customer.name,
                },
            },
        )
    )

    return new_session.id


@step(given, "a session with a thirsty customer", target_fixture="session_id")
def given_a_session_with_a_thirsty_customer(
    context: ContextOfTest,
    new_session: Session,
    customer: Customer,
) -> SessionId:
    store = context.container[SessionStore]

    context.sync_await(
        store.create_event(
            session_id=new_session.id,
            source=EventSource.CUSTOMER,
            kind=EventKind.MESSAGE,
            trace_id="<main>",
            data={
                "message": "I'm thirsty",
                "participant": {
                    "id": customer.id,
                    "display_name": customer.name,
                },
            },
        )
    )

    return new_session.id


@step(given, "a session with a few messages", target_fixture="session_id")
def given_a_session_with_a_few_messages(
    context: ContextOfTest,
    new_session: Session,
    agent: Agent,
    customer: Customer,
) -> SessionId:
    store = context.container[SessionStore]

    messages = [
        {
            "source": EventSource.CUSTOMER,
            "message": "hey there",
        },
        {
            "source": EventSource.AI_AGENT,
            "message": "Hi, how can I help you today?",
        },
        {
            "source": EventSource.CUSTOMER,
            "message": "What was the first name of the famous Einstein?",
        },
    ]

    for m in messages:
        context.sync_await(
            store.create_event(
                session_id=new_session.id,
                source=m["source"] == EventSource.AI_AGENT
                and EventSource.AI_AGENT
                or EventSource.CUSTOMER,
                kind=EventKind.MESSAGE,
                trace_id="<main>",
                data={
                    "message": cast(str, m["message"]),
                    "participant": {
                        "customer": {
                            "id": customer.id,
                            "display_name": customer.name,
                        },
                        "ai_agent": {
                            "id": agent.id,
                            "display_name": agent.name,
                        },
                    }[cast(EventSource, m["source"]).value],
                },
            )
        )

    return new_session.id


@step(
    given,
    parsers.parse("a tool event with data, {tool_event_data}"),
    target_fixture="session_id",
)
def given_a_session_with_tool_event(
    context: ContextOfTest,
    session_id: SessionId,
    tool_event_data: str,
) -> SessionId:
    store = context.container[SessionStore]
    session = context.sync_await(store.read_session(session_id=session_id))

    context.sync_await(
        store.create_event(
            session_id=session.id,
            source=EventSource.AI_AGENT,
            kind=EventKind.TOOL,
            trace_id="<main>",
            data=json.loads(tool_event_data),
        )
    )

    return session.id
