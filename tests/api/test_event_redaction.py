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
from typing import cast

from parlant.api.sessions import (
    REDACTED_EVENT_DATA,
    REDACTED_REASONING_LABEL,
    event_to_dto,
)
from parlant.core import metadata as metadata_utils
from parlant.core.common import JSONSerializable
from parlant.core.sessions import Event, EventId, EventKind, EventSource


def _make_event(kind: EventKind, data: JSONSerializable) -> Event:
    now = datetime.now(timezone.utc)
    return Event(
        id=EventId("evt-1"),
        source=EventSource.AI_AGENT,
        kind=kind,
        creation_utc=now,
        modified_utc=now,
        offset=0,
        trace_id="trace-1",
        data=data,
        metadata={},
        deleted=False,
    )


def _status_event(
    *,
    message: str | None = "Considering the customer's request",
    inner_data: JSONSerializable | None = None,
    status: str = "processing",
) -> Event:
    data: dict[str, JSONSerializable] = {"status": status}
    if message is not None:
        data["message"] = message
    if inner_data is not None:
        data["data"] = inner_data
    return _make_event(EventKind.STATUS, data)


def _internal_reasoning_event(
    *,
    message: str = "The customer is asking about order 123; I should check the warehouse first.",
    chunks: list[str | None] | None = None,
) -> Event:
    """A status event carrying the model's own chain of thought, flagged the way
    the engine flags it (``__parlant__.internal_reasoning``)."""
    event = _status_event(message=message, inner_data={"stage": "generating"})
    data = cast(dict[str, JSONSerializable], event.data)
    if chunks is not None:
        data["chunks"] = cast(JSONSerializable, chunks)
    return Event(
        id=event.id,
        source=event.source,
        kind=event.kind,
        creation_utc=event.creation_utc,
        modified_utc=event.modified_utc,
        offset=event.offset,
        trace_id=event.trace_id,
        data=data,
        metadata=metadata_utils.set_internal_value({}, ("internal_reasoning",), True),
        deleted=False,
    )


def _tool_event() -> Event:
    return _make_event(
        EventKind.TOOL,
        {
            "tool_calls": [
                {
                    "tool_id": "svc:search",
                    "rationale": "the customer asked to search",
                    "arguments": {"query": "sedans"},
                    "result": {
                        "data": {"records": ["internal-record-1", "internal-record-2"]},
                        "metadata": {"latency_ms": 42},
                        "control": {},
                        "canned_responses": [],
                        "canned_response_fields": {},
                    },
                }
            ]
        },
    )


def test_that_internal_reasoning_is_replaced_by_a_thinking_label_when_reasoning_is_not_authorized() -> (
    None
):
    event = _internal_reasoning_event(
        chunks=[
            "The customer is asking",
            " about order 123;",
            " I should check the warehouse first.",
            None,
        ],
    )

    dto = event_to_dto(event, redact_reasoning=True)

    data = cast(dict[str, JSONSerializable], dto.data)
    # Nothing of the chain of thought survives in any field: the client gets the
    # same shape the engine's own "Thinking" stage label has.
    assert data["message"] == REDACTED_REASONING_LABEL == "Thinking"
    assert "chunks" not in data
    # The status value itself and any operational data survive — a client still
    # knows the agent is working and can render progress.
    assert data["status"] == "processing"
    assert data["data"] == {"stage": "generating"}


def test_that_internal_reasoning_still_streaming_is_replaced_without_its_chunks() -> None:
    event = _internal_reasoning_event(chunks=["The customer is asking", " about order 123;"])

    dto = event_to_dto(event, redact_reasoning=True)

    data = cast(dict[str, JSONSerializable], dto.data)
    assert data["message"] == REDACTED_REASONING_LABEL
    assert "chunks" not in data


def test_that_internal_reasoning_is_returned_verbatim_when_reasoning_is_authorized() -> None:
    event = _internal_reasoning_event(chunks=["The customer is asking", None])

    dto = event_to_dto(event, redact_reasoning=False)

    assert dto.data == event.data


def test_that_a_tool_narration_passes_through_when_reasoning_is_not_authorized() -> None:
    # A narration is written by the integrator for the customer to read; it is
    # not the model's reasoning, so no permission gates it.
    event = _status_event(message="Checking with the warehouse…", inner_data=None)

    dto = event_to_dto(event, redact_reasoning=True)

    assert dto.data == {"status": "processing", "message": "Checking with the warehouse…"}


def test_that_a_built_in_stage_label_passes_through_when_reasoning_is_not_authorized() -> None:
    event = _status_event(message="Running tools", inner_data={"stage": "tools"})

    dto = event_to_dto(event, redact_reasoning=True)

    assert dto.data == {
        "status": "processing",
        "message": "Running tools",
        "data": {"stage": "tools"},
    }


def test_that_a_status_event_without_a_message_is_unchanged_when_redacted() -> None:
    event = _status_event(message=None, inner_data={"stage": "completed"})

    dto = event_to_dto(event, redact_reasoning=True)

    data = cast(dict[str, JSONSerializable], dto.data)
    assert "message" not in data
    assert data["status"] == "processing"
    assert data["data"] == {"stage": "completed"}


def test_that_tool_result_data_is_stubbed_when_tool_data_is_not_authorized() -> None:
    event = _tool_event()

    dto = event_to_dto(event, redact_tool_data=True)

    data = cast(dict[str, JSONSerializable], dto.data)
    call = cast(list[dict[str, JSONSerializable]], data["tool_calls"])[0]
    result = cast(dict[str, JSONSerializable], call["result"])
    assert result["data"] == REDACTED_EVENT_DATA
    # Everything around the raw data stays: the metadata, rationale and arguments
    # are operationally useful and not the sensitive payload.
    assert result["metadata"] == {"latency_ms": 42}
    assert call["rationale"] == "the customer asked to search"
    assert call["arguments"] == {"query": "sedans"}


def test_that_events_are_returned_verbatim_when_authorized() -> None:
    status_event = _status_event(message="internal narration", inner_data={"stage": "generating"})
    tool_event = _tool_event()

    status_dto = event_to_dto(status_event)
    tool_dto = event_to_dto(tool_event)

    assert status_dto.data == status_event.data
    assert tool_dto.data == tool_event.data


def test_that_the_reasoning_flag_does_not_redact_tool_result_data() -> None:
    event = _tool_event()

    dto = event_to_dto(event, redact_reasoning=True, redact_tool_data=False)

    data = cast(dict[str, JSONSerializable], dto.data)
    call = cast(list[dict[str, JSONSerializable]], data["tool_calls"])[0]
    result = cast(dict[str, JSONSerializable], call["result"])
    assert result["data"] == {"records": ["internal-record-1", "internal-record-2"]}


def test_that_the_tool_data_flag_does_not_redact_the_status_message() -> None:
    event = _status_event(message="internal narration")

    dto = event_to_dto(event, redact_reasoning=False, redact_tool_data=True)

    data = cast(dict[str, JSONSerializable], dto.data)
    assert data["message"] == "internal narration"


def test_that_a_message_event_is_never_redacted() -> None:
    event = _make_event(
        EventKind.MESSAGE,
        {
            "message": "Here's what I found.",
            "participant": {"display_name": "Agent"},
            "draft": "internal draft reasoning",
        },
    )

    dto = event_to_dto(event, redact_reasoning=True, redact_tool_data=True)

    assert dto.data == event.data


def test_that_redaction_does_not_mutate_the_stored_event() -> None:
    event = _status_event(message="internal narration")

    event_to_dto(event, redact_reasoning=True)

    # The persisted event keeps its full message — redaction is exposure-only.
    data = cast(dict[str, JSONSerializable], event.data)
    assert data["message"] == "internal narration"


def _event_with_metadata(
    kind: EventKind, data: JSONSerializable, metadata: JSONSerializable
) -> Event:
    now = datetime.now(timezone.utc)
    return Event(
        id=EventId("evt-2"),
        source=EventSource.AI_AGENT,
        kind=kind,
        creation_utc=now,
        modified_utc=now,
        offset=0,
        trace_id="trace-1",
        data=data,
        metadata=cast(dict[str, JSONSerializable], metadata),
        deleted=False,
    )


def test_that_internal_event_metadata_is_stripped_when_not_authorized() -> None:
    event = _event_with_metadata(
        EventKind.STATUS,
        {"status": "ready"},
        {"__parlant__": {"diagnostic_report_uri": "/debug/reports/trace-1"}, "public": "ok"},
    )

    dto = event_to_dto(event, redact_internal_metadata=True)

    assert dto.metadata == {"public": "ok"}


def test_that_internal_event_metadata_is_returned_verbatim_when_authorized() -> None:
    event = _event_with_metadata(
        EventKind.STATUS,
        {"status": "ready"},
        {"__parlant__": {"diagnostic_report_uri": "/debug/reports/trace-1"}, "public": "ok"},
    )

    dto = event_to_dto(event, redact_internal_metadata=False)

    assert dto.metadata == event.metadata


def test_that_tool_result_internal_metadata_is_stripped_when_not_authorized() -> None:
    event = _make_event(
        EventKind.TOOL,
        {
            "tool_calls": [
                {
                    "tool_id": "svc:search",
                    "result": {
                        "data": {"records": []},
                        "metadata": {
                            "latency_ms": 42,
                            "__parlant__": {"error_details": "Traceback... secret path"},
                        },
                    },
                }
            ]
        },
    )

    dto = event_to_dto(event, redact_internal_metadata=True)

    data = cast(dict[str, JSONSerializable], dto.data)
    tool_call = cast(list[dict[str, JSONSerializable]], data["tool_calls"])[0]
    result = cast(dict[str, JSONSerializable], tool_call["result"])
    assert result["metadata"] == {"latency_ms": 42}


def test_that_an_error_status_exception_is_stubbed_when_not_authorized() -> None:
    event = _make_event(
        EventKind.STATUS,
        {"status": "error", "data": {"exception": "Traceback (most recent call last): ... secret"}},
    )

    dto = event_to_dto(event, redact_internal_metadata=True)

    data = cast(dict[str, JSONSerializable], dto.data)
    inner = cast(dict[str, JSONSerializable], data["data"])
    assert "secret" not in cast(str, inner["exception"])


def test_that_an_error_status_exception_is_returned_when_authorized() -> None:
    event = _make_event(
        EventKind.STATUS,
        {"status": "error", "data": {"exception": "Traceback: boom"}},
    )

    dto = event_to_dto(event, redact_internal_metadata=False)

    data = cast(dict[str, JSONSerializable], dto.data)
    inner = cast(dict[str, JSONSerializable], data["data"])
    assert inner["exception"] == "Traceback: boom"
