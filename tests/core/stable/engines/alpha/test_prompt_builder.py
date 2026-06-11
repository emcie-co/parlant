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

from parlant.core.common import Criticality, generate_id
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.tools import Tool, ToolId, ToolOverlap


def _tool(name: str, description: str, *, consequential: bool = False) -> Tool:
    return Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description,
        metadata={},
        parameters={},
        required=[],
        consequential=consequential,
        overlap=ToolOverlap.NONE,
    )


def _match(
    condition: str, action: str, criticality: Criticality = Criticality.MEDIUM
) -> GuidelineMatch:
    now = datetime.now(timezone.utc)
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        content=GuidelineContent(condition=condition, action=action),
        enabled=True,
        tags=[],
        metadata={},
        criticality=criticality,
    )
    return GuidelineMatch(guideline=guideline, rationale="because")


# ───────────────────── guideline instructions vs. list ──────────────────────


def test_that_guideline_instructions_explain_how_to_follow_without_listing_guidelines() -> None:
    prompt = PromptBuilder().add_guideline_instructions().build()

    assert "RELEVANT DOMAIN PROTOCOL GUIDELINES" in prompt
    assert "You may choose not to follow a guideline only" in prompt
    # The explanation must not contain any of the actual matched guidelines.
    assert "Guideline #" not in prompt


def test_that_matched_guidelines_list_the_guidelines_without_the_explanation() -> None:
    match = _match("the customer asks about toppings", "list the available toppings")
    representations = {match.guideline.id: internal_representation(match.guideline)}

    prompt = PromptBuilder().add_matched_guidelines([match], {}, representations).build()

    assert "Guideline #1)" in prompt
    assert "list the available toppings" in prompt
    # The how/when explanation belongs to add_guideline_instructions, not here.
    assert "You may choose not to follow a guideline only" not in prompt


def test_that_matched_guidelines_lead_with_a_skip_if_already_satisfied_rule() -> None:
    # Co-located with the list (turn-level), so the anti-repetition rule has the
    # same recency as the guidelines themselves rather than living far up in the
    # cached system block.
    match = _match("the customer asks about toppings", "list the available toppings")
    representations = {match.guideline.id: internal_representation(match.guideline)}

    prompt = PromptBuilder().add_matched_guidelines([match], {}, representations).build()

    assert "ALREADY satisfied" in prompt
    assert "skip it silently" in prompt
    # The assessment must be internal — no narrated "let me check the guidelines" preamble.
    assert "This whole assessment is INTERNAL" in prompt


def test_that_matched_guidelines_renders_an_empty_state_when_there_are_no_matches() -> None:
    prompt = PromptBuilder().add_matched_guidelines([], {}, {}).build()

    assert "Guideline #" not in prompt
    assert "No special behavioral guidelines" in prompt


def test_that_matched_guidelines_list_their_associated_tools() -> None:
    match = _match("the customer asks about the weather", "tell them the forecast")
    representations = {match.guideline.id: internal_representation(match.guideline)}
    tool_enabled = {match: [ToolId(service_name="local", tool_name="get_weather")]}

    prompt = PromptBuilder().add_matched_guidelines([], tool_enabled, representations).build()

    assert "tell them the forecast" in prompt
    assert "get_weather" in prompt
    assert "consider using" in prompt.lower()


def test_that_matched_low_criticality_guidelines_list_their_associated_tools() -> None:
    match = _match("the customer greets you", "greet back", criticality=Criticality.LOW)
    representations = {match.guideline.id: internal_representation(match.guideline)}
    tool_enabled = {match: [ToolId(service_name="local", tool_name="say_hello")]}

    prompt = (
        PromptBuilder()
        .add_matched_low_criticality_guidelines([], tool_enabled, representations)
        .build()
    )

    assert "greet back" in prompt
    assert "say_hello" in prompt
    assert "consider using" in prompt.lower()


# ──────────────── low-criticality instructions vs. list ─────────────────────


def test_that_low_criticality_instructions_explain_without_listing() -> None:
    prompt = PromptBuilder().add_low_criticality_guideline_instructions().build()

    assert "general principles" in prompt
    assert "you may ignore" in prompt.lower()
    assert "When always, then" not in prompt


def test_that_matched_low_criticality_guidelines_list_the_principles() -> None:
    match = _match(
        "the customer is chatty",
        "keep it brief",
        criticality=Criticality.LOW,
    )
    representations = {match.guideline.id: internal_representation(match.guideline)}

    prompt = (
        PromptBuilder().add_matched_low_criticality_guidelines([match], {}, representations).build()
    )

    assert "keep it brief" in prompt


# ─────────────────────────── tool descriptions ──────────────────────────────


def test_that_tool_descriptions_list_name_and_description_as_optional() -> None:
    prompt = (
        PromptBuilder()
        .add_tool_descriptions([_tool("get_weather", "Get the current weather for a city.")])
        .build()
    )

    assert "AVAILABLE TOOLS" in prompt
    assert "get_weather: Get the current weather for a city." in prompt
    # Framed as optional — the agent may use them but doesn't have to.
    assert "MAY use the following tools" in prompt
    assert "NOT required" in prompt


def test_that_consequential_tools_carry_a_caution_note() -> None:
    prompt = (
        PromptBuilder()
        .add_tool_descriptions(
            [
                _tool("get_weather", "Get the weather."),
                _tool("charge_card", "Charge the customer's card.", consequential=True),
            ]
        )
        .build()
    )

    # The consequential note attaches only to the consequential tool.
    assert "CONSEQUENTIAL" in prompt
    assert "confirm with the user" in prompt
    charge_line = next(line for line in prompt.splitlines() if "charge_card" in line)
    weather_line = next(line for line in prompt.splitlines() if "get_weather" in line)
    assert "CONSEQUENTIAL" in charge_line
    assert "CONSEQUENTIAL" not in weather_line


def test_that_tool_descriptions_render_an_empty_state_when_there_are_no_tools() -> None:
    prompt = PromptBuilder().add_tool_descriptions([]).build()

    assert "No tools should be used" in prompt
    assert "AVAILABLE TOOLS" not in prompt
