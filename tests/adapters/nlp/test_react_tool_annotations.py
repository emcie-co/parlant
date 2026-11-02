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

"""Live verification that tool-parameter annotations reach and steer the model.

The Compass tool-call trigger is the ReAct generator's native function schema.
These tests run real generations and assert that ToolParameterOptions actually
change the tool call — the only way to prove the metadata is transmitted and
used, not merely composed into a string. They skip when GEMINI_API_KEY is absent.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any, Sequence

import google.genai
import pytest

from parlant.core.container import Container

from parlant.adapters.nlp.gemini_service import GeminiReactGenerator
from parlant.core.loggers import Logger
from parlant.core.nlp.react import Message, Role, TextPart, tool_specs_from_tools
from parlant.core.tools import Tool, ToolOverlap, ToolParameterDescriptor, ToolParameterOptions

_LIVE_MODEL = "gemini-3.7-flash"


def _require_gemini() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set; skipping live tool-annotation test")


def _tool(
    name: str,
    description: str,
    parameters: dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]],
    required: list[str],
) -> Tool:
    return Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description,
        metadata={},
        parameters=parameters,
        required=required,
        consequential=False,
        overlap=ToolOverlap.NONE,
    )


async def _step(container: Container, tool: Tool, user_text: str, *, tool_choice: Any) -> Any:
    generator = GeminiReactGenerator(
        model=_LIVE_MODEL,
        logger=container[Logger],
        client=google.genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
    )
    return await generator.step(
        [Message(role=Role.USER, parts=[TextPart(text=user_text)])],
        tool_specs_from_tools([tool]),
        tool_choice=tool_choice,
    )


def _tool_calls(result: Any, name: str) -> Sequence[Any]:
    return [call for call in result.tool_calls if call.name == name]


async def test_that_live_examples_steer_the_argument_format(container: Container) -> None:
    _require_gemini()

    tool = _tool(
        "book_slot",
        "Book a calendar slot for the customer.",
        {
            "slot": (
                # No format in the description — only the example carries it.
                {
                    "type": "string",
                    "description": "The requested time slot.",
                    "examples": ["Fri 14:00"],
                },
                ToolParameterOptions(),
            )
        },
        ["slot"],
    )

    result = await _step(
        container, tool, "Book me a slot for Monday at 9am.", tool_choice={"name": "book_slot"}
    )

    calls = _tool_calls(result, "book_slot")
    assert calls, "expected book_slot to be called"
    slot = str(calls[0].args["slot"])
    # The example ("Fri 14:00": 3-letter day + 24h time) is a format the model
    # would not default to; matching it proves the example reached the model.
    assert re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{2}:\d{2}$", slot), slot


async def test_that_live_significance_disambiguates_the_value(container: Container) -> None:
    _require_gemini()

    tool = _tool(
        "charge",
        "Charge the customer's card.",
        {
            "amount": (
                {"type": "integer", "description": "The amount to charge."},
                ToolParameterOptions(
                    significance="the amount to charge, expressed in cents (not dollars)"
                ),
            )
        },
        ["amount"],
    )

    result = await _step(
        container, tool, "Please charge the customer five dollars.", tool_choice={"name": "charge"}
    )

    calls = _tool_calls(result, "charge")
    assert calls, "expected charge to be called"
    # Without the significance hint the model would pass 5; the "in cents" note
    # steers it to 500.
    assert int(calls[0].args["amount"]) == 500, calls[0].args


async def test_that_live_customer_source_prevents_fabrication(container: Container) -> None:
    _require_gemini()

    tool = _tool(
        "redeem_coupon",
        "Redeem a discount coupon on the customer's order.",
        {
            "code": (
                {"type": "string", "description": "The coupon code."},
                ToolParameterOptions(source="customer"),
            )
        },
        ["code"],
    )

    # The customer never provided a code; source="customer" tells the model to
    # ask rather than invent one, so it should NOT call the tool this turn.
    result = await _step(
        container,
        tool,
        "I'd like to redeem a coupon on my order, please.",
        tool_choice="auto",
    )

    assert not _tool_calls(result, "redeem_coupon"), (
        "the model fabricated a coupon code instead of asking the customer"
    )


async def test_that_live_hidden_parameter_is_filled_internally(container: Container) -> None:
    _require_gemini()

    tool = _tool(
        "file_report",
        "File an internal bug report about a problem the customer described.",
        {
            "summary": (
                {"type": "string", "description": "A short summary of the problem."},
                ToolParameterOptions(),
            ),
            "internal_severity": (
                {
                    "type": "string",
                    "description": "Severity: one of low/medium/high.",
                    "enum": ["low", "medium", "high"],
                },
                # Hidden: the model should fill it itself, never asking the customer.
                ToolParameterOptions(hidden=True),
            ),
        },
        ["summary", "internal_severity"],
    )

    result = await _step(
        container,
        tool,
        "The checkout page crashes every time I try to pay — this is really urgent.",
        tool_choice={"name": "file_report"},
    )

    calls = _tool_calls(result, "file_report")
    assert calls, "expected file_report to be called"
    # The model supplies the hidden, required parameter on its own (treating it as
    # its own knowledge) rather than leaving it empty or refusing.
    assert calls[0].args.get("internal_severity") in {"low", "medium", "high"}, calls[0].args
