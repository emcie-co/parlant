import json
import os
from typing import Any, Literal, Sequence, cast
from openai import AsyncClient

from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.sessions import Event, ToolEventData
from emcie.server.engines.event_emitter import EmittedEvent


def make_llm_client(provider: Literal["openai", "together"]) -> AsyncClient:
    if provider == "openai":
        return AsyncClient(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "together":
        return AsyncClient(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )


def events_to_json(events: Sequence[Event]) -> str:
    event_dicts = [event_to_dict(e) for e in events]
    return json.dumps(event_dicts)


def event_to_dict(event: Event) -> dict[str, Any]:
    return {
        "id": event.id,
        "kind": event.kind,
        "source": {
            "client": "user",
            "server": "assistant",
        }.get(event.source),
        "data": event.data,
    }


def context_variables_to_json(
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
) -> str:
    context_values = {
        variable.name: {
            "description": variable.description,
            "value": value.data,
        }
        for variable, value in context_variables
    }

    return json.dumps(context_values)


def emitted_tool_events_to_dicts(
    events: Sequence[EmittedEvent],
) -> list[dict[str, Any]]:
    return [emitted_tool_event_to_dict(e) for e in events]


def emitted_tool_event_to_dict(event: EmittedEvent) -> dict[str, Any]:
    assert event.kind == "tool"

    return {
        "kind": event.kind,
        "data": cast(ToolEventData, event.data)["tool_calls"],
    }
