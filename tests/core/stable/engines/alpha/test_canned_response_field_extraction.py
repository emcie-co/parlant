"""Tests for canned response field extraction methods."""

from typing import Any
from unittest.mock import Mock

import pytest

from parlant.core.engines.alpha.canned_response_generator import (
    CannedResponseContext,
    ToolBasedFieldExtraction,
)
from parlant.core.sessions import EventKind, ToolCall, ToolEventData, ToolResult


@pytest.fixture
def tool_based_extractor() -> ToolBasedFieldExtraction:
    """Create a ToolBasedFieldExtraction instance."""
    return ToolBasedFieldExtraction()


def create_context_with_tool_result(
    field_name: str,
    field_value: Any,
) -> CannedResponseContext:
    """Helper to create a CannedResponseContext with a tool call result."""
    tool_call: ToolCall = {
        "tool_id": "test_tool",
        "arguments": {},
        "result": ToolResult(
            data={},
            metadata={},
            control={},
            canned_responses=[],
            canned_response_fields={
                field_name: field_value,
            },
        ),
    }

    tool_event_data: ToolEventData = {"tool_calls": [tool_call]}

    event = Mock()
    event.kind = EventKind.TOOL
    event.data = tool_event_data

    context = Mock(spec=CannedResponseContext)
    context.interaction_history = [event]
    context.staged_tool_events = []

    return context  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_that_tool_based_field_extraction_returns_integer_zero_value(
    tool_based_extractor: ToolBasedFieldExtraction,
) -> None:
    """Test that integer zero (0) is correctly extracted as a valid field value."""
    context = create_context_with_tool_result("result_count", 0)

    found, value = await tool_based_extractor.extract(
        canned_response="Test response",
        field_name="result_count",
        context=context,
    )

    assert found is True
    assert value == 0


@pytest.mark.asyncio
async def test_that_tool_based_field_extraction_returns_false_boolean_value(
    tool_based_extractor: ToolBasedFieldExtraction,
) -> None:
    """Test that boolean False is correctly extracted as a valid field value."""
    context = create_context_with_tool_result("is_available", False)

    found, value = await tool_based_extractor.extract(
        canned_response="Test response",
        field_name="is_available",
        context=context,
    )

    assert found is True
    assert value is False


@pytest.mark.asyncio
async def test_that_tool_based_field_extraction_returns_empty_string_value(
    tool_based_extractor: ToolBasedFieldExtraction,
) -> None:
    """Test that empty string is correctly extracted as a valid field value."""
    context = create_context_with_tool_result("description", "")

    found, value = await tool_based_extractor.extract(
        canned_response="Test response",
        field_name="description",
        context=context,
    )

    assert found is True
    assert value == ""


@pytest.mark.asyncio
async def test_that_tool_based_field_extraction_returns_empty_list_value(
    tool_based_extractor: ToolBasedFieldExtraction,
) -> None:
    """Test that empty list is correctly extracted as a valid field value."""
    context = create_context_with_tool_result("items", [])

    found, value = await tool_based_extractor.extract(
        canned_response="Test response",
        field_name="items",
        context=context,
    )

    assert found is True
    assert value == []


@pytest.mark.asyncio
async def test_that_tool_based_field_extraction_returns_empty_dict_value(
    tool_based_extractor: ToolBasedFieldExtraction,
) -> None:
    """Test that empty dict is correctly extracted as a valid field value."""
    context = create_context_with_tool_result("metadata", {})

    found, value = await tool_based_extractor.extract(
        canned_response="Test response",
        field_name="metadata",
        context=context,
    )

    assert found is True
    assert value == {}


@pytest.mark.asyncio
async def test_that_tool_based_field_extraction_returns_none_when_field_not_found(
    tool_based_extractor: ToolBasedFieldExtraction,
) -> None:
    """Test that None is returned when the field is not found."""
    context = create_context_with_tool_result("other_field", "some_value")

    found, value = await tool_based_extractor.extract(
        canned_response="Test response",
        field_name="nonexistent_field",
        context=context,
    )

    assert found is False
    assert value is None
