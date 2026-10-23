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

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from parlant.core.agents import (
    Agent,
    AgentId,
    AgentUpdateParams,
    Effort,
    MessageOutputMode,
    Moderation,
)
from parlant.core.app_modules.agents import AgentGroupUpdateParamsModel, AgentModule
from parlant.core.common import ItemNotFoundError, JSONSerializable
from parlant.core.groups import GroupId
from parlant.core.nlp.model_routing import (
    ModelConfig,
    ModelRouteValue,
    SecretRef,
    SecretValue,
)
from parlant.core.nlp.react import ReasoningConfig
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


_PUBLIC_MODERATIONS: Mapping[str, Moderation] = MappingProxyType(
    {moderation.value: moderation for moderation in Moderation}
)
_PUBLIC_MODERATION_VALUES: Mapping[Moderation, str] = MappingProxyType(
    {moderation: value for value, moderation in _PUBLIC_MODERATIONS.items()}
)
_PUBLIC_MESSAGE_OUTPUT_MODES: Mapping[str, MessageOutputMode] = MappingProxyType(
    {
        "block": MessageOutputMode.BLOCK,
        "stream": MessageOutputMode.STREAM,
        "voice": MessageOutputMode.VOICE,
    }
)
_PUBLIC_MESSAGE_OUTPUT_MODE_VALUES: Mapping[MessageOutputMode, str] = MappingProxyType(
    {mode: value for value, mode in _PUBLIC_MESSAGE_OUTPUT_MODES.items()}
)
_PUBLIC_EFFORTS: Mapping[str, Effort] = MappingProxyType(
    {
        "min": Effort.MIN,
        "low": Effort.LOW,
        "medium": Effort.MEDIUM,
        "high": Effort.HIGH,
        "max": Effort.MAX,
    }
)
_PUBLIC_EFFORT_VALUES: Mapping[Effort, str] = MappingProxyType(
    {effort: value for value, effort in _PUBLIC_EFFORTS.items()}
)


class AgentTunnelHandlers:
    def __init__(self, agent_module: AgentModule) -> None:
        self._agent_module = agent_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "agents.create": self.create,
                "agents.list": self.list,
                "agents.retrieve": self.retrieve,
                "agents.update": self.update,
                "agents.delete": self.delete,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        name = _require_agent_name(params.get("name"))
        agent_id = _parse_agent_id(params.get("id"), field="id")
        instructions = _parse_instructions(params.get("instructions"))
        max_steps = _parse_max_steps(params.get("max_steps"))
        groups = _parse_group_ids(params.get("groups"), field="groups")

        agent = await self._agent_module.create(
            id=agent_id,
            name=name,
            instructions=instructions,
            max_steps=max_steps,
            message_output_mode=_parse_message_output_mode(params.get("message_output_mode")),
            effort=_parse_effort(params.get("effort")),
            groups=groups,
            moderation=_parse_moderation(params.get("moderation")),
            message_review=_parse_bool(params.get("message_review"), field="message_review")
            or False,
            tool_review=_parse_bool(params.get("tool_review"), field="tool_review") or False,
        )
        return {"agent": await self._serialize_agent(agent)}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        agents = await self._agent_module.find()
        return {"agents": await asyncio.gather(*(self._serialize_agent(agent) for agent in agents))}

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        agent_id = _require_agent_id(params.get("agent_id"), field="agent_id")
        try:
            agent = await self._agent_module.read(agent_id=agent_id)
        except ItemNotFoundError:
            return {"agent": None}
        return {"agent": await self._serialize_agent(agent)}

    async def update(self, params: dict[str, Any]) -> dict[str, Any]:
        agent_id = _require_agent_id(params.get("agent_id"), field="agent_id")
        update_params: AgentUpdateParams = {}
        if "name" in params:
            name = _parse_optional_agent_name(params["name"])
            if name is not None:
                update_params["name"] = name
        if "instructions" in params:
            update_params["instructions"] = _parse_instructions(params["instructions"])
        if "max_steps" in params:
            max_steps = _parse_max_steps(params["max_steps"])
            if max_steps is not None:
                update_params["max_steps"] = max_steps
        if "message_output_mode" in params:
            message_output_mode = _parse_message_output_mode(params["message_output_mode"])
            if message_output_mode is not None:
                update_params["message_output_mode"] = message_output_mode
        if "effort" in params:
            effort = _parse_effort(params["effort"])
            if effort is not None:
                update_params["effort"] = effort

        group_params = params.get("groups")
        if group_params is not None and not isinstance(group_params, dict):
            raise ValueError("groups must be an object or null")
        groups = None
        if group_params is not None:
            groups = AgentGroupUpdateParamsModel(
                add=_parse_group_ids(group_params.get("add"), field="groups.add"),
                remove=_parse_group_ids(group_params.get("remove"), field="groups.remove"),
            )

        agent = await self._agent_module.update(
            agent_id=agent_id,
            name=update_params.get("name"),
            instructions=update_params.get("instructions"),
            max_steps=update_params.get("max_steps"),
            message_output_mode=update_params.get("message_output_mode"),
            effort=update_params.get("effort"),
            groups=groups,
            moderation=_parse_moderation(params.get("moderation")),
            message_review=_parse_bool(params.get("message_review"), field="message_review"),
            tool_review=_parse_bool(params.get("tool_review"), field="tool_review"),
        )
        return {"agent": await self._serialize_agent(agent)}

    async def delete(self, params: dict[str, Any]) -> dict[str, Any]:
        agent_id = _require_agent_id(params.get("agent_id"), field="agent_id")
        await self._agent_module.delete(agent_id=agent_id)
        return {}

    async def _serialize_agent(self, agent: Agent) -> dict[str, Any]:
        models = await self._agent_module.read_models(agent.id)
        return _serialize_agent(agent, models)


def _require_agent_name(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 100:
        raise ValueError("Agent name must be a string between 1 and 100 characters")
    return value


def _parse_optional_agent_name(value: object) -> str | None:
    if value is None:
        return None
    return _require_agent_name(value)


def _parse_agent_id(value: object, *, field: str) -> AgentId | None:
    if value is None:
        return None
    return _require_agent_id(value, field=field)


def _require_agent_id(value: object, *, field: str) -> AgentId:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return AgentId(value)


def _parse_instructions(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Agent instructions must be a string or null")
    return value


def _parse_max_steps(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ValueError("max_steps must be an integer greater than or equal to 1")
    return value


def _parse_group_ids(value: object, *, field: str) -> list[GroupId] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(group_id, str) for group_id in value):
        raise ValueError(f"{field} must be a list of strings or null")
    return [GroupId(group_id) for group_id in value]


def _parse_moderation(value: object) -> Moderation | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _PUBLIC_MODERATIONS:
        raise ValueError(f"moderation must be one of {sorted(_PUBLIC_MODERATIONS)}")
    return _PUBLIC_MODERATIONS[value]


def _parse_bool(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")
    return value


def _parse_message_output_mode(value: object) -> MessageOutputMode | None:
    if value is None:
        return None
    return _require_message_output_mode(value)


def _require_message_output_mode(value: object) -> MessageOutputMode:
    if not isinstance(value, str) or value not in _PUBLIC_MESSAGE_OUTPUT_MODES:
        raise ValueError(f"Unsupported message output mode: {value}")
    return _PUBLIC_MESSAGE_OUTPUT_MODES[value]


def _parse_effort(value: object) -> Effort | None:
    if value is None:
        return None
    return _require_effort(value)


def _require_effort(value: object) -> Effort:
    if not isinstance(value, str) or value not in _PUBLIC_EFFORTS:
        raise ValueError(f"Unsupported effort: {value}")
    return _PUBLIC_EFFORTS[value]


def _serialize_agent(
    agent: Agent,
    models: Mapping[str, ModelRouteValue],
) -> dict[str, Any]:
    return {
        "id": str(agent.id),
        "name": agent.name,
        "instructions": agent.instructions,
        "max_steps": agent.max_steps,
        "message_output_mode": _PUBLIC_MESSAGE_OUTPUT_MODE_VALUES[agent.message_output_mode],
        "effort": _PUBLIC_EFFORT_VALUES[agent.effort],
        "groups": [str(group_id) for group_id in agent.groups],
        "creation_utc": agent.creation_utc.isoformat(),
        "modified_utc": agent.modified_utc.isoformat(),
        "engine": agent.engine,
        "moderation": _PUBLIC_MODERATION_VALUES[agent.moderation],
        "message_review": agent.message_review,
        "tool_review": agent.tool_review,
        "models": _serialize_model_routes(models),
    }


def _serialize_model_routes(
    routes: Mapping[str, ModelRouteValue],
) -> dict[str, str | dict[str, Any]]:
    result: dict[str, str | dict[str, Any]] = {}
    for route, target in routes.items():
        if isinstance(target, str):
            result[route] = target
        elif isinstance(target, ModelConfig):
            # A function provider has no static model string; show its name so the
            # route is still visible rather than silently dropped.
            model = (
                target.model
                if isinstance(target.model, str)
                else getattr(target.model, "__name__", "<provider>")
            )
            result[route] = {
                "model": model,
                "args": {
                    name: _redact_model_argument(value) for name, value in target.args.items()
                },
                "rate_limit": (
                    {
                        "max_concurrency": target.rate_limit.max_concurrency,
                        "requests_per_minute": target.rate_limit.requests_per_minute,
                        "bucket": target.rate_limit.bucket,
                    }
                    if target.rate_limit
                    else None
                ),
                # Per-route reasoning depth (ReAct/schematic). A str effort is
                # normalized to ReasoningConfig in ModelConfig.__post_init__.
                "reasoning": (
                    {"effort": target.reasoning.effort}
                    if isinstance(target.reasoning, ReasoningConfig)
                    else None
                ),
            }
    return result


def _redact_model_argument(value: Any) -> JSONSerializable:
    # Redaction is by TYPE, not by name: a secret must be a SecretRef or
    # SecretValue to be hidden. This relies on the developer-facing contract
    # that secrets are always typed — a credential written as a plain string is
    # a contract violation and will be exposed verbatim to the (admin-tier)
    # Dashboard. Do NOT "downgrade" this to pass plain args through as safe;
    # if untyped secrets ever become possible, redact by name here instead.
    if isinstance(value, SecretRef):
        return {"$secret": "[REDACTED]"}
    if isinstance(value, SecretValue):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(key): _redact_model_argument(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_model_argument(item) for item in value]
    return cast(JSONSerializable, value)
