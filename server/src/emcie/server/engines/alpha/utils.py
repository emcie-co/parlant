from contextlib import contextmanager
import json
import os
import time
from typing import Any, Literal, Sequence, cast
from loguru import logger
from openai import AsyncClient

from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.sessions import Event, ToolEventData
from emcie.server.engines.common import ProducedEvent


def make_llm_client(provider: Literal["openai", "together"]) -> AsyncClient:
    if provider == "openai":
        return AsyncClient(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "together":
        return AsyncClient(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )


@contextmanager
def duration_logger(operation_name: str) -> Any:
    t_start = time.time()

    try:
        yield
    finally:
        t_end = time.time()
        logger.debug(f"{operation_name} took {round(t_end - t_start, 3)}s")


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


def produced_tool_events_to_dicts(
    produced_events: Sequence[ProducedEvent],
) -> list[dict[str, Any]]:
    return [produced_tool_event_to_dict(e) for e in produced_events]


def produced_tool_event_to_dict(produced_event: ProducedEvent) -> dict[str, Any]:
    assert produced_event.kind == Event.TOOL_KIND

    return {
        "kind": produced_event.kind,
        "data": cast(ToolEventData, produced_event.data)["tool_results"],
    }
