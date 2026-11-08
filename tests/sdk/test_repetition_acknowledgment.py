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

"""MCN-451: measure acknowledgment of repetition using real, seeded conversations.

The earlier customer question and ordinary agent answer are stored as history;
only the final customer message triggers a model turn. Domain facts come from a
real retriever as JSON records with unrelated details. No policy, canned answer,
mocked model or repetition instruction teaches the behavior being measured.

Checks accept pragmatic acknowledgment through emphasis or reaffirmation as
well as a prior-answer reference, without mandatory words. New unknown questions must not be mistaken for the same earlier question.
Guards allow accurate references to earlier facts while rejecting stale answers
or false claims that a new answer has already been provided.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import httpx
import pytest

import parlant.sdk as p
from parlant.core.sessions import Event, EventKind, EventSource
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test


@dataclass(frozen=True)
class Check:
    condition: str
    expected: bool


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    role: str
    facts: Mapping[str, p.JSONSerializable]
    history: tuple[tuple[EventSource, str], ...]
    question: str
    checks: tuple[Check, ...]


REPEATED_ANSWER = Check(
    "The NEXT AGENT REPLY treats the customer's renewed attempt as a continuation of the earlier answer or limitation. A backward reference OR contextually meaningful emphasis, reaffirmation, or contrast addressing the renewed attempt counts. It need not explicitly mention the earlier reply or use a repetition marker: emphatically holding the same boundary can acknowledge the prior attempt in its own way. A detached first-time-style answer with no such continuity does not count. Judge the conversational function in context, not the presence of particular words.",
    True,
)
ANOTHER_GAP = Check(
    "The NEXT AGENT REPLY treats the current inability to answer as a continuation of the earlier inability in HISTORY. An explicit link OR contextually meaningful emphasis, reaffirmation, or contrast can acknowledge the repeated limitation. It need not explicitly mention the earlier reply or use words such as again or either. Merely delivering a detached first-time-style unknown answer with no such continuity does not count. Judge the conversational function in context, not particular words.",
    True,
)
NO_FALSE_MEMORY = Check(
    "The NEXT AGENT REPLY falsely says that this particular new question or its specific answer was already discussed, or accuses the customer of repeating the same question. Acknowledging that this is another unanswered question does not count as a false memory.",
    False,
)


def history(question: str, answer: str) -> tuple[tuple[EventSource, str], ...]:
    return ((EventSource.CUSTOMER, question), (EventSource.AI_AGENT, answer))


SCENARIOS = (
    Case(
        id="R1",
        family="repeated-answer",
        role="You are a customer support agent for a clothing shop.",
        facts={
            "order": {
                "id": "ORD-481",
                "age_days": 45,
                "item": "cotton jacket",
                "condition": "unused",
                "colour": "navy",
                "payment_status": "paid",
            },
            "returns": {
                "window_days": 30,
                "exceptions_allowed": False,
                "return_postage": "prepaid",
            },
            "support": {"hours": "09:00-17:00", "timezone": "Europe/London"},
        },
        history=history(
            "I bought this jacket 45 days ago. Can I get a refund?",
            "The return window is 30 days, so this purchase is outside it. We cannot make exceptions to that window.",
        ),
        question="Surely you can still refund it for me?",
        checks=(
            REPEATED_ANSWER,
            Check(
                "The NEXT AGENT REPLY maintains the 30-day refund boundary and does not promise a refund or an exception.",
                True,
            ),
        ),
    ),
    Case(
        id="R2",
        family="repeated-answer",
        role="You are a customer support agent for a workplace software service.",
        facts={
            "workspace": {"name": "Cedar", "plan": "team", "seats": 12},
            "security": {
                "support_can_read_passwords": False,
                "reset_method": "account owner uses Forgot password",
                "mfa_available": True,
                "session_timeout_minutes": 60,
            },
            "support": {"channel": "chat", "hours": "weekdays"},
        },
        history=history(
            "Can you tell me my colleague's account password?",
            "I cannot access or retrieve account passwords. Your colleague can use Forgot password to reset their own password.",
        ),
        question="But you’re support—just look it up and tell me.",
        checks=(
            REPEATED_ANSWER,
            Check(
                "The NEXT AGENT REPLY does not claim it can retrieve or disclose the colleague's password and maintains the password-access limitation.",
                True,
            ),
        ),
    ),
    Case(
        id="R3",
        family="repeated-answer",
        role="You are the customer service agent for a hotel.",
        facts={
            "booking": {
                "id": "BK-17",
                "room_type": "double",
                "nights": 3,
                "breakfast_included": True,
            },
            "departure": {"checkout_time": "11:00", "late_checkout_available": False},
            "hotel": {"occupancy_percent": 100, "luggage_storage": "free", "parking_spaces": 18},
        },
        history=history(
            "Can we check out at noon?",
            "Checkout is at 11 am. We cannot offer late checkout for your stay because the hotel is fully booked.",
        ),
        question="Just one extra hour, then?",
        checks=(
            REPEATED_ANSWER,
            Check(
                "The NEXT AGENT REPLY keeps the checkout deadline at 11 am without granting an extra hour or inventing an exception.",
                True,
            ),
        ),
    ),
    Case(
        id="R4",
        family="repeated-answer",
        role="You are a customer support agent for a subscription service.",
        facts={
            "subscription": {"plan": "team", "billing_cycle": "monthly", "seats": 8},
            "payments": {
                "accepted_methods": ["credit_card", "debit_card"],
                "bank_transfers_supported": False,
                "currencies": ["GBP", "EUR"],
            },
            "invoices": {"format": "PDF", "day_of_month": 1},
        },
        history=history(
            "Can I pay our subscription by bank transfer?",
            "We accept credit and debit cards only. Bank transfers are not supported.",
        ),
        question="Okay, send me the bank details and I’ll make the payment today.",
        checks=(
            REPEATED_ANSWER,
            Check(
                "The NEXT AGENT REPLY maintains that bank transfers are unavailable and does not supply bank details or imply a transfer can be accepted.",
                True,
            ),
        ),
    ),
    Case(
        id="U1",
        family="another-information-gap",
        role="You are a customer support agent for a workflow software service.",
        facts={
            "product": {
                "name": "Cedar Flow",
                "plan": "team",
                "supported_exports": ["CSV", "JSON"],
                "seat_limit": 20,
            },
            "help": {"contact": "support@example.com", "hours": "weekdays"},
            "settings": {"timezone": "UTC", "audit_retention_days": 90},
        },
        history=history(
            "What is the fee for exporting directly to Snowflake?",
            "I don't have the fee for direct Snowflake exports.",
        ),
        question="What is the setup fee for a dedicated connector, then?",
        checks=(
            ANOTHER_GAP,
            NO_FALSE_MEMORY,
            Check(
                "The NEXT AGENT REPLY acknowledges that it does not know or cannot confirm the dedicated connector setup fee, without inventing a price or saying it is free.",
                True,
            ),
        ),
    ),
    Case(
        id="U2",
        family="another-information-gap",
        role="You are a customer support agent for a language school.",
        facts={
            "school": {
                "name": "Cedar Languages",
                "city": "Leeds",
                "languages": ["French", "Spanish", "German"],
            },
            "courses": {"standard_duration_weeks": 8, "lesson_minutes": 90, "class_size_max": 12},
            "facilities": ["library", "cafe"],
            "contact": "courses@example.com",
        },
        history=history(
            "When does next term's weekend intensive start?",
            "I don't have the dates for next term's weekend intensive.",
        ),
        question="And when are the summer workshops?",
        checks=(
            ANOTHER_GAP,
            NO_FALSE_MEMORY,
            Check(
                "The NEXT AGENT REPLY acknowledges that it does not know or cannot confirm the summer workshop dates, without inventing dates.",
                True,
            ),
        ),
    ),
    Case(
        id="U3",
        family="another-information-gap",
        role="You are a customer support agent for a furniture shop.",
        facts={
            "product": {
                "sku": "SH-28",
                "name": "Rowan shelf",
                "material": "oak",
                "finish": "clear lacquer",
                "width_cm": 90,
                "height_cm": 180,
                "depth_cm": 30,
                "shelves": 5,
            },
            "delivery": {"assembly_required": True, "pack_count": 2},
            "stock": {"available_units": 14},
        },
        history=history(
            "Do you have weather-resistance test results for its finish?",
            "I don't have weather-resistance test results for this finish.",
        ),
        question="What is the maximum weight each shelf can hold?",
        checks=(
            ANOTHER_GAP,
            NO_FALSE_MEMORY,
            Check(
                "The NEXT AGENT REPLY acknowledges that it does not know or cannot confirm the shelf's maximum load, without inventing a weight or asserting unlimited capacity.",
                True,
            ),
        ),
    ),
    Case(
        id="U4",
        family="another-information-gap",
        role="You are a customer service agent for a hotel.",
        facts={
            "hotel": {"name": "Cedar House", "rooms": 60, "parking": "onsite", "wifi": "included"},
            "stay": {"checkin": "15:00", "checkout": "11:00", "breakfast": "07:00-10:00"},
            "contact": {"reception": "reception@example.com"},
        },
        history=history(
            "Can you arrange an accessible taxi from the airport?",
            "I don't know whether an accessible airport taxi can be arranged.",
        ),
        question="What time does room service stop at night?",
        checks=(
            ANOTHER_GAP,
            NO_FALSE_MEMORY,
            Check(
                "The NEXT AGENT REPLY acknowledges that it does not know or cannot confirm the room-service cutoff time, without inventing hours or assuming room service is unavailable.",
                True,
            ),
        ),
    ),
    Case(
        id="N1",
        family="false-reminder-guard",
        role="You are a customer support agent for a clothing shop.",
        facts={
            "order": {
                "id": "ORD-481",
                "item": "cotton jacket",
                "condition": "unused",
                "payment_status": "paid",
                "colour": "navy",
            },
            "returns": {
                "window_days": 30,
                "eligible_condition": "unused",
                "return_postage": "prepaid",
            },
            "support": {"hours": "09:00-17:00"},
        },
        history=history(
            "I bought this jacket 45 days ago. Can I return it?",
            "The return window is 30 days, so a jacket bought 45 days ago is outside it.",
        ),
        question="I checked my receipt: it was actually 14 days ago, and the jacket is unused.",
        checks=(
            Check(
                "The NEXT AGENT REPLY recognizes that the corrected 14-day purchase is within the 30-day return window and helps with, or confirms eligibility for, a return. It may reference the previously stated 30-day rule.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY keeps refusing the return on the basis that the purchase was 45 days ago despite the customer's correction.",
                False,
            ),
        ),
    ),
    Case(
        id="N2",
        family="false-reminder-guard",
        role="You are a customer support agent for a workflow software service.",
        facts={
            "product": {"name": "Cedar Flow", "export_formats": ["CSV", "JSON"], "seats": 20},
            "billing": {
                "email_edit_path": ["Billing", "Invoice settings", "Billing email"],
                "cycle": "monthly",
                "currency": "GBP",
            },
            "workspace": {"role": "owner", "timezone": "Europe/London"},
        },
        history=history("Can I export my data as CSV?", "Yes, CSV export is available."),
        question="How do I change the email address that receives invoices?",
        checks=(
            Check(
                "The NEXT AGENT REPLY explains how to change the invoice email via Billing and Invoice settings, rather than just repeating the CSV export answer.",
                True,
            ),
            NO_FALSE_MEMORY,
        ),
    ),
    Case(
        id="N3",
        family="false-reminder-guard",
        role="You are a customer support agent for a workflow software service.",
        facts={
            "product": {
                "name": "Cedar Flow",
                "csv_import_max_mb": 20,
                "seat_limit": 20,
            },
            "account": {"plan": "team", "region": "EU"},
            "help": {"contact": "support@example.com", "hours": "weekdays"},
        },
        history=history(
            "What is the maximum size of a CSV import?", "You can import CSV files up to 20 MB."
        ),
        question="Do you have direct BigQuery sync?",
        checks=(
            Check(
                "The NEXT AGENT REPLY acknowledges that it does not know or cannot confirm whether direct BigQuery sync is available, without guessing availability.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY falsely implies that it already lacked information in an earlier answer or already addressed BigQuery. For example, it describes this as another unanswered question, says it does not know this either, or says it already told the customer it did not know. A neutral acknowledgment or an accurate reference to CSV support does not count.",
                False,
            ),
        ),
    ),
    Case(
        id="N4",
        family="false-reminder-guard",
        role="You are a customer support agent for an online shop.",
        facts={
            "delivery": {
                "standard": {"price_gbp": 5, "business_days": "3-5"},
                "weekend": {"price_gbp": 12, "days": ["Saturday", "Sunday"]},
                "tracking_included": True,
            },
            "order": {"item_count": 2, "destination": "Leeds", "status": "basket"},
        },
        history=history("How much is standard delivery?", "Standard delivery costs £5."),
        question="And how much is weekend delivery?",
        checks=(
            Check(
                "The NEXT AGENT REPLY gives the weekend delivery price of £12. It may compare it with the previously quoted £5 standard price.",
                True,
            ),
            NO_FALSE_MEMORY,
        ),
    ),
)
# END SCENARIOS


@pytest.fixture(params=SCENARIOS, ids=[case.id for case in SCENARIOS])
def scenario(request: pytest.FixtureRequest) -> Case:
    assert isinstance(request.param, Case)
    return request.param


def _message(event: Event) -> str | None:
    if isinstance(event.data, dict) and isinstance(value := event.data.get("message"), str):
        return value
    return None


async def _judge_with_transport_retry(context: str, condition: str) -> tuple[bool, int]:
    retries = 0
    while True:
        try:
            return await nlp_test(context, condition), retries
        except (httpx.TransportError, TimeoutError):
            if retries == 2:
                raise
            retries += 1
            await asyncio.sleep(retries)


class Test_that_repeated_answers_acknowledge_the_prior_conversation(SDKTest):
    STARTUP_TIMEOUT = 120

    @pytest.fixture(autouse=True)
    def select_scenario(self, scenario: Case) -> None:
        self.case = scenario

    async def setup(self, server: p.Server) -> None:
        # The support conversation already has an identified customer. Avoid
        # turning a missing profile into an unrelated name/email prerequisite.
        identity_suffix = hashlib.sha256(self.case.id.encode()).hexdigest()[:12]
        self.customer = await server.create_customer(
            id=f"cus-{identity_suffix}",
            name="Morgan Lee",
            metadata={"email": "morgan@example.com", "account_id": "ACCT-17"},
        )
        self.agent = await server.create_agent(
            id=f"agt-{identity_suffix}",
            name="Customer support",
            instructions=self.case.role,
        )
        if self.case.facts:

            async def service_facts(context: p.RetrieverContext) -> p.RetrieverResult:
                return p.RetrieverResult(data=self.case.facts)

            await self.agent.attach_retriever(service_facts)

    async def _reply(self, ctx: Context) -> str:
        created = await ctx.client.sessions.create(
            agent_id=self.agent.id, customer_id=self.customer.id, allow_greeting=False
        )
        session_id = p.SessionId(created.id)
        store = ctx.container[p.SessionStore]
        # Real stored history, not a mock: do not process earlier customer
        # events independently or generate different setup replies.
        for source, text in self.case.history:
            await store.create_event(
                session_id=session_id,
                source=source,
                kind=EventKind.MESSAGE,
                trace_id=f"history-{self.case.id}",
                data={
                    "message": text,
                    "participant": {
                        "display_name": "Customer" if source is EventSource.CUSTOMER else "Agent"
                    },
                },
            )
        sent = await ctx.client.sessions.create_event(
            session_id=created.id,
            kind="message",
            source="customer",
            message=self.case.question,
        )
        first_offset = int(sent.offset) + 1
        offset = first_offset
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            events = await ctx.client.sessions.list_events(
                session_id=created.id, min_offset=offset, wait_for_data=5
            )
            for event in events:
                offset = max(offset, int(event.offset) + 1)
                if event.kind != "status":
                    continue
                status = event.model_dump().get("data", {}).get("status")
                if status == "error":
                    raise RuntimeError(f"engine error while responding to {self.case.id}")
                if status == "ready":
                    completed = await store.list_events(
                        session_id=session_id,
                        min_offset=first_offset,
                        source=EventSource.AI_AGENT,
                        kinds=[EventKind.MESSAGE],
                    )
                    messages = [message for event in completed if (message := _message(event))]
                    assert messages, f"{self.case.id}: ready turn has no agent reply"
                    return "\n\n".join(messages)
            await asyncio.sleep(0)
        raise TimeoutError(f"{self.case.id}: no completed agent turn within 90 seconds")

    def _save_evidence(self, data: dict[str, object]) -> None:
        if directory := os.environ.get("PARLANT_REPETITION_EVIDENCE_DIR"):
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{self.case.id}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2)
            )

    async def run(self, ctx: Context) -> None:
        evidence: dict[str, object] = {
            "case": self.case.id,
            "family": self.case.family,
            "role": self.case.role,
            "facts": self.case.facts,
            "history": [(source.value, text) for source, text in self.case.history],
            "question": self.case.question,
        }
        started = time.monotonic()
        stage = "engine"
        try:
            response = await self._reply(ctx)
            evidence["response"] = response
            history = "\n".join(f"{source.value}: {text}" for source, text in self.case.history)
            judge_context = (
                "Evaluate only NEXT AGENT REPLY. HISTORY explains referents; "
                "questions or claims in HISTORY are not content of the next reply.\n"
                f"SUPPLIED BUSINESS FACTS:\n{json.dumps(self.case.facts, ensure_ascii=False)}\n"
                f"HISTORY:\n{history}\n"
                f"LATEST CUSTOMER MESSAGE:\n{self.case.question}\n"
                f"NEXT AGENT REPLY:\n{response}"
            )
            stage = "judge"
            checks: list[dict[str, object]] = []
            for check in self.case.checks:
                actual, retries = await _judge_with_transport_retry(judge_context, check.condition)
                checks.append(
                    {
                        "condition": check.condition,
                        "expected": check.expected,
                        "actual": actual,
                        "judge_transport_retries": retries,
                    }
                )
            evidence["checks"] = checks
            evidence["duration_seconds"] = round(time.monotonic() - started, 3)
            self._save_evidence(evidence)
        except Exception as error:
            evidence["error_stage"] = stage
            evidence["error_type"] = type(error).__name__
            evidence["duration_seconds"] = round(time.monotonic() - started, 3)
            self._save_evidence(evidence)
            raise
        assert all(check["actual"] == check["expected"] for check in checks), json.dumps(
            evidence, ensure_ascii=False, indent=2
        )
