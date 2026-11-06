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

"""MCN-429: default conversational judgment, exercised through the real SDK engine.

These are the twenty transcript decision points agreed on 2026-11-05. Seeded
history stops before the reply under test: example answers and later customer
clarifications never reach the agent. Only its role and factual service context
are configured. There is no disambiguation policy, canned answer, mocked model,
or replacement responder.

Retrieved context is ordinary JSON business data, not prose that explains the
ambiguity or tells the agent which distinction to make. Role descriptions name
the business domain without enumerating the competing tasks in the scenario.

NLP assertions describe conversational acts, not exact sentences, punctuation,
word counts or a prohibition on all questions. Near-misses allow ordinary task
details and diagnosis, while rejecting unnecessary meaning/priority questions.
Negative assertions execute unconditionally. Initial failures are measurements,
not permission to change the engine or relax an agreed expectation.
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


WORKSPACE_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "organization": {
        "id": "WK-17",
        "name": "Cedar Workspace",
        "category": "workspace administration",
        "plan": "team",
        "locale": "en-GB",
        "created_at": "2025-07-12T10:00:00Z",
    },
    "members": [
        {"id": "USR-17", "name": "Morgan Lee", "role": "owner", "status": "active"},
        {
            "id": "USR-21",
            "name": "Alex",
            "role": "administrator",
            "status": "active",
            "permissions": ["members:read", "billing:read", "billing:write"],
        },
        {"id": "USR-25", "name": "Leah", "role": "viewer", "status": "active"},
        {
            "id": "USR-32",
            "name": "Sam",
            "role": "administrator",
            "status": "active",
            "permissions": ["billing:read", "billing:write", "reports:read"],
        },
    ],
    "settings": {"digest_frequency": "weekly", "storage_used_mb": 380, "seat_limit": 10},
    "service_catalog": [
        "member_invitations",
        "member_access_changes",
        "workspace_roles",
        "invoice_queries",
        "profile_updates",
        "audit_logs",
        "notification_settings",
    ],
}

APPLICATION_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "product": {
        "name": "Cedar Office",
        "category": "business software",
        "version": "2026.11.2",
        "released_at": "2026-11-05T07:00:00Z",
    },
    "account": {"id": "ACCT-17", "plan": "team", "locale": "en-GB", "seats": 5},
    "preferences": {"date_format": "DD/MM/YYYY", "number_format": "1,234.56"},
    "recent_jobs": [
        {"id": "JOB-101", "type": "report_export", "status": "completed", "rows": 180},
        {"id": "JOB-102", "type": "spreadsheet_import", "status": "completed", "rows": 450},
        {"id": "JOB-103", "type": "contact_sync", "status": "completed", "rows": 24},
    ],
    "job_snapshot_at": "2026-11-04T16:30:00Z",
    "service_catalog": [
        "sign_in",
        "spreadsheet_imports",
        "data_exports",
        "reports",
        "billing",
        "notifications",
        "integrations",
        "user_profiles",
    ],
}

RETAIL_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "store": {"name": "Cedar Store", "currency": "GBP", "locale": "en-GB"},
    "customer": {"id": "ACCT-17", "name": "Morgan Lee", "loyalty_points": 340},
    "recent_orders": [
        {
            "id": "4700",
            "items": [{"sku": "ESPRESSO-01", "name": "Espresso machine", "quantity": 1}],
            "status": "delivered",
            "delivered_at": "2026-11-01T12:30:00Z",
            "return_eligible_until": "2026-11-30",
            "total": 379,
        },
        {
            "id": "4610",
            "items": [{"sku": "KITCHEN-SCALE", "name": "Kitchen scale", "quantity": 1}],
            "status": "delivered",
            "delivered_at": "2026-10-15T10:20:00Z",
            "total": 29,
        },
    ],
    "subscriptions": [
        {
            "id": "SUB-51",
            "product": "Dog food",
            "status": "active",
            "interval_days": 30,
            "delivery_address": {"line_1": "8 Oak Lane", "city": "Cambridge"},
            "quantity": 2,
        },
        {"id": "SUB-77", "product": "Coffee beans", "status": "paused", "interval_days": 30},
    ],
    "service_catalog": [
        "order_status",
        "order_updates",
        "returns",
        "subscriptions",
        "payment_queries",
        "loyalty_points",
        "account_profiles",
        "product_information",
    ],
}

TRAVEL_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "organization": {"name": "Cedar Travel", "category": "corporate travel"},
    "traveler": {
        "id": "TRV-17",
        "name": "Morgan Lee",
        "home_airport": "LHR",
        "seat_preference": "aisle",
        "cost_center": "Operations",
    },
    "recent_trips": [
        {
            "id": "TRIP-80",
            "destination": "Berlin",
            "ended_at": "2026-10-18",
            "status": "completed",
            "hotel": {"id": "HOTEL-8", "total": 260, "currency": "GBP"},
        },
        {"id": "TRIP-62", "destination": "Dublin", "ended_at": "2026-09-11", "status": "completed"},
    ],
    "billing": {"account_id": "ACCT-17", "currency": "GBP", "receipt_format": "PDF"},
    "service_catalog": [
        "flight_bookings",
        "itinerary_changes",
        "expense_claims",
        "itemised_receipts",
        "travel_policy",
        "traveler_profiles",
        "airport_transfers",
        "accessibility_requests",
    ],
}

EDUCATION_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "organization": {"name": "Cedar Learning", "category": "education provider"},
    "account": {"id": "ACCT-17", "name": "Morgan Lee", "locale": "en-GB"},
    "students": [
        {"id": "STU-17", "name": "Morgan Lee", "portal_status": "active"},
        {"id": "STU-34", "name": "Jess Lee", "relationship": "daughter", "portal_status": "active"},
    ],
    "enrolments": [
        {
            "id": "ENR-208",
            "student_id": "STU-17",
            "course_id": "SPN-101",
            "group": "evening",
            "status": "active",
        },
    ],
    "course_catalog": [
        {"id": "SPN-101", "title": "Spanish 1", "groups": ["evening", "weekend"], "fee": 220},
        {"id": "ART-201", "title": "Drawing 2", "groups": ["weekday"], "fee": 180},
        {"id": "CMP-110", "title": "Computer skills", "groups": ["weekday", "weekend"], "fee": 150},
    ],
    "term": {"name": "Autumn 2026", "ends_at": "2026-12-18", "campus": "Central"},
    "service_catalog": [
        "course_enrolment",
        "enrolment_changes",
        "student_portal_access",
        "certificates",
        "fee_queries",
        "attendance",
        "learning_support",
        "student_records",
    ],
}

ORDER_CONTEXT: Mapping[str, p.JSONSerializable] = {
    **RETAIL_CONTEXT,
    "order": {
        "id": "4821",
        "customer_id": "ACCT-17",
        "created_at": "2026-11-05T11:20:00Z",
        "status": "processing",
        "currency": "GBP",
        "items": [
            {"sku": "GRINDER-01", "quantity": 1, "unit_price": 129.0},
            {"sku": "FILTER-100", "quantity": 1, "unit_price": 12.0},
        ],
        "subtotal": 141.0,
        "tax": 28.2,
        "delivery_fee": 4.99,
        "total": 174.19,
        "billing_address": {
            "name": "Morgan Lee",
            "line_1": "14 Maple Road",
            "city": "Cambridge",
            "postal_code": "CB1 2AB",
            "country": "GB",
        },
        "delivery_method": "standard",
        "delivery_address": {
            "name": "Morgan Lee",
            "line_1": "8 Oak Lane",
            "city": "Cambridge",
            "postal_code": "CB2 1CD",
            "country": "GB",
        },
        "payment": {"method": "card", "last_four": "4242", "status": "captured"},
        "customer_note": None,
    },
}

APPOINTMENT_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "generated_at": "2026-11-09T09:00:00Z",
    "time_zone": "Europe/London",
    "customer": {"id": "ACCT-17", "name": "Morgan Lee", "preferred_contact": "email"},
    "locations": [
        {"id": "LOC-2", "name": "Central", "parking": "street", "accessibility": "step_free"},
        {"id": "LOC-3", "name": "North", "parking": "onsite", "accessibility": "step_free"},
    ],
    "preferences": {"email_reminders": True, "sms_reminders": False, "locale": "en-GB"},
    "account_balance": {"currency": "GBP", "amount": 0},
    "appointments": [
        {
            "id": "APPT-4821",
            "customer_id": "ACCT-17",
            "start_at": "2026-11-10T14:00:00Z",
            "end_at": "2026-11-10T14:30:00Z",
            "status": "confirmed",
            "service": "consultation",
            "location_id": "LOC-2",
            "duration_minutes": 30,
            "staff_id": "STAFF-8",
            "confirmation_code": "CONF-2871",
        }
    ],
    "api_reference": {
        "/appointments": {"methods": ["GET", "POST"]},
        "/appointments/{id}": {
            "methods": ["GET", "PATCH", "DELETE"],
            "patch_fields": ["start_at", "location_id", "notes"],
        },
        "/locations": {"methods": ["GET"]},
    },
}

FITNESS_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "generated_at": "2026-11-09T09:00:00Z",
    "account": {"id": "TEAM-17", "name": "Cedar Team", "manager": "Morgan Lee"},
    "billing_preferences": {"currency": "GBP", "invoice_day": 23, "paperless": True},
    "facilities": ["studio", "pool", "lockers", "cycle_racks"],
    "guest_passes_remaining": 4,
    "attendance_summary": {"last_month_visits": 18, "favorite_location": "Central"},
    "membership": {
        "id": "MEM-17",
        "plan": "team_monthly",
        "status": "active",
        "renews_at": "2026-11-23",
        "member_count": 6,
        "auto_renew": True,
        "freeze_days_remaining": 14,
    },
    "class_bookings": [
        {
            "id": "CLS-YOGA",
            "class_name": "Yoga",
            "start_at": "2026-11-16T18:00:00Z",
            "status": "confirmed",
            "seats": 6,
            "instructor": "Taylor",
            "location": "Studio 2",
        },
        {
            "id": "CLS-PILATES",
            "class_name": "Pilates",
            "start_at": "2026-11-17T18:00:00Z",
            "status": "confirmed",
            "seats": 6,
            "instructor": "Jordan",
            "location": "Studio 1",
        },
    ],
    "credit_balance": 12,
    "time_zone": "Europe/London",
    "service_catalog": [
        "membership_changes",
        "class_booking_changes",
        "payment_queries",
        "facilities",
        "guest_passes",
        "team_administration",
    ],
}

BILLING_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "preferences": {"locale": "en-GB", "invoice_delivery": "email", "time_zone": "Europe/London"},
    "usage": {"active_seats": 4, "api_calls": 1204, "storage_mb": 380},
    "payment_method": {"type": "card", "last_four": "4242", "expires": "2028-04"},
    "service_catalog": [
        "team_setup",
        "single_sign_on",
        "billing",
        "security_settings",
        "usage_reports",
        "integrations",
        "notifications",
    ],
    "account": {
        "id": "ACCT-17",
        "owner": "Morgan Lee",
        "email": "morgan@example.com",
        "plan": "team",
        "seats": 5,
        "status": "active",
    },
    "transactions": [
        {
            "id": "INV-70",
            "plan": "team",
            "period_start": "2026-10-01",
            "period_end": "2026-11-01",
            "amount": 89,
            "currency": "USD",
            "posted_at": "2026-10-02T08:00:00Z",
            "status": "settled",
        },
        {
            "id": "INV-71",
            "plan": "team",
            "period_start": "2026-11-01",
            "period_end": "2026-12-01",
            "amount": 89,
            "currency": "USD",
            "posted_at": "2026-11-02T08:00:00Z",
            "status": "settled",
        },
        {
            "id": "INV-72",
            "plan": "team",
            "period_start": "2026-11-01",
            "period_end": "2026-12-01",
            "amount": 89,
            "currency": "USD",
            "posted_at": "2026-11-02T08:04:00Z",
            "status": "settled",
        },
    ],
    "refunds": [],
}

WEBSITE_CONTEXT: Mapping[str, p.JSONSerializable] = {
    "organization": {"name": "Parlant", "support_hours": "weekdays", "language": "en"},
    "documentation_categories": ["setup", "SDK", "tools", "hosting", "observability"],
    "site_metadata": {"default_locale": "en", "theme": "light", "updated_at": "2026-11-05"},
    "products": [
        {
            "name": "Parlant",
            "category": "customer-facing agent framework",
            "capabilities": ["conversations", "business rules", "tool integrations"],
        },
        {"name": "Zendesk", "category": "helpdesk platform", "features": ["tickets", "bots"]},
    ],
    "site_links": [
        {"label": "Documentation", "url": "https://example.com/docs"},
        {"label": "Pricing", "url": "https://example.com/pricing"},
        {
            "label": "Contact",
            "url": "https://example.com/contact",
            "topics": ["sales", "partnerships", "support"],
        },
        {"label": "Careers", "url": "https://example.com/careers"},
        {"label": "About", "url": "https://example.com/about"},
        {"label": "Status", "url": "https://example.com/status"},
    ],
}


# BEGIN SCENARIOS
SCENARIOS: tuple[Case, ...] = (
    Case(
        id="A1",
        family="unclear-meaning",
        role="You are a customer support agent for a workspace administration service.",
        facts=WORKSPACE_CONTEXT,
        history=(
            (EventSource.CUSTOMER, "Alex and Sam both have access to our billing workspace."),
        ),
        question="Remove his access.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks the customer to identify which person they mean, rather than selecting a person itself.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY claims or commits to removing Alex’s or Sam’s access without first getting the target identified.",
                False,
            ),
        ),
    ),
    Case(
        id="A2",
        family="unclear-meaning",
        role="You are a customer support agent for an online store.",
        facts=ORDER_CONTEXT,
        history=(),
        question="I need to change the address on order 4821.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks which type or purpose of address the customer wants changed, such as billing versus delivery.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY treats delivery or billing as already chosen and proceeds specifically with that field without checking which is intended.",
                False,
            ),
        ),
    ),
    Case(
        id="A3",
        family="unclear-meaning",
        role="You are a customer support agent for a business software company.",
        facts=APPLICATION_CONTEXT,
        history=(),
        question="It doesn’t work after the update.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks for context about what the customer is trying to do or what is not working, instead of assuming a particular activity.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY diagnoses or gives a specific fix for an assumed activity, such as a login failure, without learning what the customer is doing.",
                False,
            ),
        ),
    ),
    Case(
        id="A4",
        family="unclear-meaning",
        role="You are a customer support agent for an appointment-based business.",
        facts=APPOINTMENT_CONTEXT,
        history=(),
        question="Move my appointment to next Friday.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks the customer to identify the intended calendar date or which Friday they mean.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY settles on a particular Friday as the intended date without checking it with the customer.",
                False,
            ),
        ),
    ),
    Case(
        id="A5",
        family="unclear-meaning",
        role="You are a customer support agent for a fitness studio.",
        facts=FITNESS_CONTEXT,
        history=(),
        question="Cancel everything for our team from next week.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks what the cancellation should include, distinguishing class bookings from the team membership or asking whether both are intended.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY takes the membership or only the class bookings as the already-settled cancellation scope without checking what everything includes.",
                False,
            ),
        ),
    ),
    Case(
        id="B1",
        family="unclear-focus",
        role="You are a customer support agent for an online store.",
        facts=RETAIL_CONTEXT,
        history=(
            (EventSource.CUSTOMER, "I need to return the espresso machine I bought last week."),
        ),
        question="I need to change the delivery address for my dog-food subscription.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks which of the return and subscription-address tasks the customer wants to handle now or first.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY proceeds into one or both task workflows without asking the customer to settle the immediate focus.",
                False,
            ),
        ),
    ),
    Case(
        id="B2",
        family="unclear-focus",
        role="You are a customer support agent for a team software service.",
        facts=BILLING_CONTEXT,
        history=(
            (EventSource.CUSTOMER, "Help me set up single sign-on for our team."),
            (EventSource.AI_AGENT, "Which identity provider does your team use?"),
        ),
        question="We were charged twice this month. Help me get that refunded.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks whether to pause or continue sign-on setup versus handling the billing issue now.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks only for billing or sign-on details, treating the immediate task as settled instead of checking the customer’s focus.",
                False,
            ),
        ),
    ),
    Case(
        id="B3",
        family="unclear-focus",
        role="You are a customer support agent for a corporate travel service.",
        facts=TRAVEL_CONTEXT,
        history=((EventSource.CUSTOMER, "Help me book a flight to Lisbon for next week."),),
        question="I need to claim back a hotel bill from last month’s Berlin trip.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks which to work on now or first: the Lisbon flight booking or the Berlin hotel-expense claim.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY starts collecting booking or expense details without checking which task the customer wants to focus on.",
                False,
            ),
        ),
    ),
    Case(
        id="B4",
        family="unclear-focus",
        role="You are a customer support agent for an education provider.",
        facts=EDUCATION_CONTEXT,
        history=(
            (EventSource.CUSTOMER, "Help me switch from the evening course to the weekend course."),
        ),
        question="My daughter also needs help resetting her student-portal password.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks whether to start with the customer’s course change or the daughter’s portal-access problem.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY merges the two people’s needs or starts one workflow without checking which task to handle now.",
                False,
            ),
        ),
    ),
    Case(
        id="B5",
        family="unclear-focus",
        role="You are the website advisor for Parlant.",
        facts=WEBSITE_CONTEXT,
        history=(
            (
                EventSource.CUSTOMER,
                "Help me work out whether Parlant could replace our Zendesk bot.",
            ),
        ),
        question="I’d like someone to contact me about a partnership.",
        checks=(
            Check(
                "The NEXT AGENT REPLY asks whether to focus now on the Zendesk product-fit discussion or the partnership contact request.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY starts product qualification or contact capture as if that workflow were already chosen, without checking the customer’s immediate focus.",
                False,
            ),
        ),
    ),
    Case(
        id="A-FP1",
        family="near-meaning",
        role="You are a customer support agent for a workspace administration service.",
        facts=WORKSPACE_CONTEXT,
        history=(
            (
                EventSource.CUSTOMER,
                "Sam is finishing up with us today. Alex is taking over his work.",
            ),
        ),
        question="Can you remove his access?",
        checks=(
            Check(
                "The NEXT AGENT REPLY offers relevant help or guidance about removing the departing person’s workspace access, or asks an ordinary task-detail or authorization question about that request. It need not repeat Sam’s name or claim a completed change.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer to choose or identify whether Alex or Sam is meant by his.",
                False,
            ),
        ),
    ),
    Case(
        id="A-FP2",
        family="near-meaning",
        role="You are a customer support agent for an online store.",
        facts=ORDER_CONTEXT,
        history=((EventSource.CUSTOMER, "I won’t be at that flat anymore when the parcel comes."),),
        question="Can we change the address?",
        checks=(
            Check(
                "The NEXT AGENT REPLY moves forward with the parcel’s destination or shipping-address change, for example by asking for the new destination.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks whether the customer means the billing address or delivery address.",
                False,
            ),
        ),
    ),
    Case(
        id="A-FP3",
        family="near-meaning",
        role="You are a customer support agent for a business software company.",
        facts=APPLICATION_CONTEXT,
        history=(
            (EventSource.CUSTOMER, "I’ve got this year’s sales in a spreadsheet."),
            (EventSource.AI_AGENT, "You can bring it in using the import button."),
        ),
        question="Yeah, it gets about halfway and just stops.",
        checks=(
            Check(
                "The NEXT AGENT REPLY continues discussing the spreadsheet import problem, for example by asking about an error or troubleshooting its stopping halfway.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer to identify what activity they are trying to do or which unrelated feature is failing, as if the import context were unknown.",
                False,
            ),
        ),
    ),
    Case(
        id="A-FP4",
        family="near-meaning",
        role="You are a customer support agent for an appointment-based business.",
        facts=APPOINTMENT_CONTEXT,
        history=(),
        question="I’m thinking of moving it to next Friday. Does that mean scrapping the whole booking and starting again?",
        checks=(
            Check(
                "The NEXT AGENT REPLY explains that rescheduling can change the existing appointment without scrapping it and starting a new booking.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer to identify which Friday or exact date next Friday means.",
                False,
            ),
        ),
    ),
    Case(
        id="A-FP5",
        family="near-meaning",
        role="You are a customer support agent for a fitness studio.",
        facts=FITNESS_CONTEXT,
        history=((EventSource.CUSTOMER, "None of us can make next week’s sessions."),),
        question="Can we cancel the whole lot?",
        checks=(
            Check(
                "The NEXT AGENT REPLY treats next week’s team class bookings as the subject and moves helpfully toward handling those bookings. It need not perform a cancellation.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks whether the customer means to cancel classes, the membership, or both.",
                False,
            ),
        ),
    ),
    Case(
        id="B-FP1",
        family="near-focus",
        role="You are a customer support agent for an online store.",
        facts=RETAIL_CONTEXT,
        history=((EventSource.CUSTOMER, "I need to return the espresso machine."),),
        question="I’ve just got the “packing your box” email for the dog food, but I’m not at that address anymore.",
        checks=(
            Check(
                "The NEXT AGENT REPLY addresses the shipment’s destination problem now, for example by asking for the new address or checking that shipment, rather than returning only to the espresso-machine return.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer to choose between handling the return and handling the shipment/address problem.",
                False,
            ),
        ),
    ),
    Case(
        id="B-FP2",
        family="near-focus",
        role="You are a customer support agent for a team software service.",
        facts=BILLING_CONTEXT,
        history=(
            (EventSource.CUSTOMER, "Help me set up single sign-on for the team."),
            (EventSource.AI_AGENT, "Which identity provider do you use?"),
        ),
        question="Hang on, why are there two charges for us on this statement?",
        checks=(
            Check(
                "The NEXT AGENT REPLY investigates or asks a useful question about the two charges on the statement. It need not resolve the billing issue immediately.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer whether to focus on billing or sign-on setup, or whether to pause one for the other.",
                False,
            ),
        ),
    ),
    Case(
        id="B-FP3",
        family="near-focus",
        role="You are a customer support agent for a corporate travel service.",
        facts=TRAVEL_CONTEXT,
        history=(),
        question="I need a Lisbon flight next week. Finance will want an itemised receipt for the claim as well.",
        checks=(
            Check(
                "The NEXT AGENT REPLY advances the Lisbon flight booking, for example by asking a travel-detail question. It need not repeat the receipt requirement in this particular reply.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer to choose between booking the flight and dealing with a separate expense or receipt task.",
                False,
            ),
        ),
    ),
    Case(
        id="B-FP4",
        family="near-focus",
        role="You are a customer support agent for an education provider.",
        facts=EDUCATION_CONTEXT,
        history=(),
        question="My daughter’s portal is driving her mad too. Anyway, can you move me to the weekend group?",
        checks=(
            Check(
                "The NEXT AGENT REPLY helps with the customer’s course change, for example by asking an enrolment or course-detail question.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer to choose between their course change and the daughter’s portal problem, or starts troubleshooting the daughter’s portal instead.",
                False,
            ),
        ),
    ),
    Case(
        id="B-FP5",
        family="near-focus",
        role="You are the website advisor for Parlant.",
        facts=WEBSITE_CONTEXT,
        history=(),
        question="Could this replace our Zendesk bot? Also, where do people go to ask about partnerships?",
        checks=(
            Check(
                "The NEXT AGENT REPLY gives the partnership contact-page pointer and advances the Zendesk fit discussion, for example with a useful question about the existing bot.",
                True,
            ),
            Check(
                "The NEXT AGENT REPLY asks the customer to choose between discussing Zendesk fit and asking about partnerships.",
                False,
            ),
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
        except httpx.TransportError:
            if retries == 2:
                raise
            retries += 1
            await asyncio.sleep(retries)


class Test_that_the_engine_disambiguates_only_when_the_customer_intent_requires_it(SDKTest):
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
        if directory := os.environ.get("PARLANT_DISAMBIGUATION_EVIDENCE_DIR"):
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
