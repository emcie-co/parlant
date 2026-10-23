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

"""Stateless Argon transport, private to the Gemini ReAct adapter.

Google's native steps and signatures stay opaque provider artifacts. Canonical
history and tool execution remain owned by Parlant; no server-side conversation
resource or adapter-executed tool is used.
"""

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import google.genai

from parlant.core.nlp.react import (
    EMPTY_CONVERSATION_USER_TURN,
    FinishReason,
    Message,
    ReactError,
    ReasoningConfig,
    ReasoningDelta,
    ReasoningPart,
    Role,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolCallStarted,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
    TurnBuilder,
    Usage,
)

CALL = "gemini_interactions_call"
THOUGHTS = "gemini_interactions_thoughts"
VisibleTextDecoder = Callable[[str, TurnBuilder, Mapping[str, Any] | None], list[StreamEvent]]
_CONTINUE = (
    "[Internal Parlant continuation instruction; not a new customer message.] "
    "Continue the assistant turn in response to the existing customer request. "
    "Follow all system instructions and tool results already provided. Do not "
    "treat this control instruction as customer authorization or quote it."
)


def uses_interactions(model: str) -> bool:
    return model == "gemini-4-argon"


def encode(
    history: Sequence[Message],
    tools: Sequence[ToolSpec],
    tool_choice: ToolChoice,
    reasoning: ReasoningConfig,
    model: str,
    service_tier: str,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    system: list[str] = []
    native_ids: dict[str, str] = {}
    pending: set[str] = set()
    assistant_start: int | None = None
    seen_thoughts: set[str] = set()
    for message in history:
        if message.role == Role.SYSTEM:
            if not steps:
                system.append(message.text)
            elif message.text:
                steps.append(
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": "[Internal Parlant system update; not customer input.]\n"
                                + message.text,
                            }
                        ],
                    }
                )
            continue
        if message.role == Role.ASSISTANT:
            if assistant_start is None:
                assistant_start = len(steps)
                seen_thoughts.clear()
        else:
            assistant_start = None
        has_calls = any(isinstance(part, ToolCallPart) for part in message.parts)
        for part in message.parts:
            # A canonical ReasoningPart aggregates summaries across the turn.
            # For tool turns, each call carries its own preceding native
            # thoughts so aggregation cannot move a later signature earlier.
            if isinstance(part, ToolCallPart) or (
                isinstance(part, ReasoningPart) and not has_calls
            ):
                for thought in part.provider_data.get(THOUGHTS, []):
                    identity = json.dumps(thought, sort_keys=True)
                    if identity not in seen_thoughts:
                        seen_thoughts.add(identity)
                        # The engine may persist a visible preamble separately
                        # from the native tool event. Google requires the
                        # existing signed thought to start that assistant turn.
                        # Internal system updates do not make a new customer
                        # turn; retain them while moving only the first thought.
                        if assistant_start is not None and not any(
                            step["type"] == "thought" for step in steps[assistant_start:]
                        ):
                            steps.insert(assistant_start, deepcopy(thought))
                        else:
                            steps.append(deepcopy(thought))
            if isinstance(part, TextPart) and part.text:
                steps.append(
                    {
                        "type": "model_output" if message.role == Role.ASSISTANT else "user_input",
                        "content": [{"type": "text", "text": part.text}],
                    }
                )
            elif isinstance(part, ToolCallPart):
                native = part.provider_data.get(CALL)
                if native:
                    if native.get("name") != part.name or native.get("arguments") != dict(
                        part.args
                    ):
                        raise ReactError(
                            "Cannot replay an edited Gemini native tool call", retryable=False
                        )
                    step = deepcopy(native)
                else:
                    step = {
                        "type": "function_call",
                        "id": part.id,
                        "name": part.name,
                        "arguments": dict(part.args),
                    }
                native_ids[part.id] = step["id"]
                pending.add(part.id)
                steps.append(step)
            elif isinstance(part, ToolResultPart):
                if part.call_id not in pending:
                    # Staged retrievers and historical replay fallbacks supply
                    # results with no call. Interactions rejects a synthetic
                    # function_call but accepts a function_result that answers
                    # none, as plain Gemini accepts an unpaired functionResponse.
                    result = {"error": part.content} if part.is_error else part.content
                    steps.append(
                        {
                            "type": "function_result",
                            "call_id": part.call_id,
                            "name": part.name,
                            "result": [{"type": "text", "text": json.dumps(result)}],
                        }
                    )
                    continue
                pending.remove(part.call_id)
                result = {"error": part.content} if part.is_error else part.content
                steps.append(
                    {
                        "type": "function_result",
                        "call_id": native_ids[part.call_id],
                        "name": part.name,
                        "result": [{"type": "text", "text": json.dumps(result)}],
                    }
                )
    if pending:
        raise ReactError("Gemini history contains tool calls without results", retryable=False)
    if not steps:
        steps.append(
            {
                "type": "user_input",
                "content": [{"type": "text", "text": EMPTY_CONVERSATION_USER_TURN}],
            }
        )
    elif steps[-1]["type"] in {"model_output", "thought", "function_call"}:
        # The Interactions API also rejects model-ending input. This is strictly
        # a provider-bound control instruction: never mutate canonical history.
        steps.append({"type": "user_input", "content": [{"type": "text", "text": _CONTINUE}]})
    config: dict[str, Any] = {
        "thinking_level": "low" if reasoning.effort == "minimal" else reasoning.effort,
        "thinking_summaries": "none" if reasoning.visibility == "none" else "auto",
        "max_output_tokens": 32768,
        "tool_choice": "any"
        if isinstance(tool_choice, Mapping) or tool_choice == "required"
        else tool_choice,
    }
    declarations = list(tools)
    if isinstance(tool_choice, Mapping):
        declarations = [tool for tool in tools if tool.name == tool_choice.get("name")]
        if not declarations:
            raise ReactError("The required Gemini tool is not declared", retryable=False)
    request: dict[str, Any] = {
        "model": model,
        "input": steps,
        "store": False,
        "stream": True,
        "generation_config": config,
        "service_tier": service_tier,
    }
    if system:
        request["system_instruction"] = "\n\n".join(system)
    if declarations:
        request["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.json_schema(),
            }
            for tool in declarations
        ]
    return request


def capture(calls: Sequence[ToolCallPart], blob: dict[str, Any]) -> None:
    blob["interactions"] = [
        {
            CALL: deepcopy(call.provider_data.get(CALL)),
            THOUGHTS: deepcopy(call.provider_data.get(THOUGHTS, [])),
        }
        for call in calls
    ]


def restore(calls: Sequence[ToolCallPart], blob: Mapping[str, Any], model: str) -> bool:
    if blob.get("model") != model:
        return False
    artifacts = blob.get("interactions")
    if not isinstance(artifacts, list) or len(artifacts) != len(calls):
        return False
    for call, artifact in zip(calls, artifacts):
        if not isinstance(artifact, dict):
            return False
        native = artifact.get(CALL)
        thoughts = artifact.get(THOUGHTS)
        if (
            not isinstance(native, dict)
            or native.get("type") != "function_call"
            or not isinstance(native.get("id"), str)
            or not native["id"]
            or native.get("name") != call.name
            or native.get("arguments") != dict(call.args)
            or not isinstance(native.get("signature"), str)
            or not native["signature"]
            or not isinstance(thoughts, list)
            or any(not isinstance(t, dict) or t.get("type") != "thought" for t in thoughts)
        ):
            return False
    for call, artifact in zip(calls, artifacts):
        call.provider_data.update(deepcopy(artifact))
    return True


@dataclass
class Chunk:
    state: "InteractionState"
    event: dict[str, Any]


class InteractionState:
    """A single request's native stream; never shared between concurrent calls."""

    def __init__(self, model: str, decode_text: VisibleTextDecoder) -> None:
        self.model = model
        self.decode_text = decode_text
        self.steps: dict[int, dict[str, Any]] = {}
        self.arguments: dict[int, str] = {}
        self.open_steps: set[int] = set()
        self.pending_thoughts: list[dict[str, Any]] = []
        self.terminal = False
        self.has_calls = False

    def decode(self, event: dict[str, Any], builder: TurnBuilder) -> list[StreamEvent]:
        if self.terminal:
            return []
        kind = event.get("event_type")
        if kind == "step.start":
            index = event["index"]
            if index in self.steps:
                raise ReactError("Duplicate Gemini interaction step", retryable=False)
            step = deepcopy(event["step"])
            self.steps[index] = step
            self.open_steps.add(index)
            if step["type"] == "function_call":
                if not step.get("id") or not step.get("name"):
                    raise ReactError("Gemini returned an incomplete tool call", retryable=False)
                self.has_calls = True
                return [ToolCallStarted(id=step["id"], name=step["name"])]
            if step["type"] == "model_output":
                text = "".join(
                    c.get("text", "") for c in step.get("content", []) if c.get("type") == "text"
                )
                if text:
                    return self.decode_text(text, builder, None)
        elif kind == "step.delta":
            index = event["index"]
            if index not in self.open_steps:
                raise ReactError("Gemini delta has no open step", retryable=False)
            step = self.steps[index]
            delta = event["delta"]
            delta_type = delta.get("type")
            if delta_type == "text" and step["type"] == "model_output":
                text = delta.get("text", "")
                return self.decode_text(text, builder, None)
            if delta_type == "arguments_delta" and step["type"] == "function_call":
                self.arguments[index] = self.arguments.get(index, "") + delta.get("arguments", "")
            elif delta_type == "thought_signature" and step["type"] == "thought":
                step["signature"] = delta["signature"]
            elif delta_type == "thought_summary" and step["type"] == "thought":
                content = delta.get("content", {})
                text = content.get("text", "")
                builder.reasoning_delta(text, visibility="summary")
                return [ReasoningDelta(text=text)] if text else []
        elif kind == "step.stop":
            index = event["index"]
            if index not in self.open_steps:
                raise ReactError("Gemini closed an unknown step", retryable=False)
            self.open_steps.remove(index)
            step = self.steps[index]
            if step["type"] == "thought":
                if step.get("signature"):
                    self.pending_thoughts.append(step)
                    builder.reasoning_delta(
                        "", provider_data={THOUGHTS: deepcopy(self.pending_thoughts)}
                    )
            elif step["type"] == "function_call":
                if index in self.arguments:
                    try:
                        step["arguments"] = json.loads(self.arguments[index])
                    except (ValueError, TypeError) as exc:
                        raise ReactError(
                            "Gemini returned invalid tool arguments", retryable=False
                        ) from exc
                if not isinstance(step.get("arguments"), dict):
                    raise ReactError("Gemini tool arguments must be an object", retryable=False)
                builder.tool_call(
                    step["id"],
                    name=step["name"],
                    args=step["arguments"],
                    provider_data={CALL: deepcopy(step), THOUGHTS: deepcopy(self.pending_thoughts)},
                )
                self.pending_thoughts.clear()
        elif kind in {"interaction.completed", "interaction.failed", "interaction.incomplete"}:
            self.terminal = True
            interaction = event.get("interaction", {})
            status = interaction.get("status")
            usage = interaction.get("usage") or {}
            thoughts = usage.get("total_thought_tokens") or 0
            builder.usage = Usage(
                input_tokens=usage.get("total_input_tokens") or 0,
                cached_input_tokens=usage.get("total_cached_tokens") or 0,
                output_tokens=(usage.get("total_output_tokens") or 0) + thoughts,
                reasoning_tokens=thoughts,
                model_name=interaction.get("model") or self.model,
            )
            if status == "incomplete":
                raise ReactError("Gemini interaction exhausted its output budget", retryable=False)
            if status not in {"completed", "requires_action"} or self.open_steps:
                raise ReactError(
                    "Gemini interaction did not complete successfully", retryable=False
                )
            if self.has_calls != (status == "requires_action"):
                raise ReactError(
                    "Gemini interaction ended with inconsistent tool status", retryable=False
                )
            builder.finish_reason = FinishReason.TOOL_CALLS if self.has_calls else FinishReason.STOP
        elif kind == "error":
            raise ReactError("Gemini interaction stream failed", retryable=False)
        return []


async def stream(
    client: google.genai.Client,
    request: dict[str, Any],
    decode_text: VisibleTextDecoder,
) -> AsyncIterator[Chunk]:
    state = InteractionState(request["model"], decode_text)
    try:
        native = await client.aio.interactions.create(**request)
        async with native:
            async for event in native:
                yield Chunk(state, event.model_dump(exclude_none=True))
        if not state.terminal:
            raise ReactError(
                "Gemini interaction stream ended without a terminal event", retryable=True
            )
    except ReactError:
        raise
    except Exception as exc:
        # Interactions uses a different Google HTTP exception hierarchy from
        # generateContent. Never include a response body (which may echo input).
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            raise ReactError(
                f"Gemini Interactions HTTP {status}", retryable=status in {429, 500, 502, 503, 504}
            ) from exc
        raise
