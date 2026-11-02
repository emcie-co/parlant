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

from typing import Any

from parlant.core.agents import AgentId
from parlant.core.app_modules.indexing import IndexingModule
from parlant.core.async_utils import Timeout
from parlant.core.indexing import (
    IndexingTask,
    IndexingTaskId,
    Invoice,
    InvoiceRuleData,
    PayloadOperation,
    RulePayload,
)
from parlant.core.rules import RuleId
from parlant.core.tunnel_handlers.common import TunnelHandlerSet


_DEFAULT_WAIT_FOR_COMPLETION_SECONDS = 60


class IndexingTunnelHandlers:
    def __init__(self, indexing_module: IndexingModule) -> None:
        self._indexing_module = indexing_module

    def handler_set(self) -> TunnelHandlerSet:
        return TunnelHandlerSet(
            unary={
                "indexing.create": self.create,
                "indexing.retrieve": self.retrieve,
            },
            streams={},
        )

    async def create(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_payloads = params.get("payloads")
        if not isinstance(raw_payloads, list) or not raw_payloads:
            raise ValueError("payloads must be a non-empty list")

        default_agent_id = _parse_optional_string(params.get("agent_id"), field="agent_id")

        payloads = [
            _parse_rule_payload(entry, default_agent_id=default_agent_id) for entry in raw_payloads
        ]

        indexing_task = await self._indexing_module.create(payloads=payloads)
        return {"indexing_task": _serialize_indexing_task(indexing_task)}

    async def retrieve(self, params: dict[str, Any]) -> dict[str, Any]:
        indexing_task_id = params.get("indexing_task_id")
        if not isinstance(indexing_task_id, str) or not indexing_task_id:
            raise ValueError("indexing_task_id must be a non-empty string")

        wait_for_completion = params.get(
            "wait_for_completion", _DEFAULT_WAIT_FOR_COMPLETION_SECONDS
        )
        if isinstance(wait_for_completion, bool) or not isinstance(wait_for_completion, int):
            raise ValueError("wait_for_completion must be an integer number of seconds")

        if wait_for_completion > 0:
            completed = await self._indexing_module.wait_for_completion(
                indexing_task_id=IndexingTaskId(indexing_task_id),
                timeout=Timeout(wait_for_completion),
            )
            if not completed:
                raise TimeoutError("Request timed out")

        indexing_task = await self._indexing_module.read(
            indexing_task_id=IndexingTaskId(indexing_task_id),
        )
        return {"indexing_task": _serialize_indexing_task(indexing_task)}


def _parse_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _parse_bool(value: object, *, field: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _parse_rule_payload(entry: object, *, default_agent_id: str | None) -> RulePayload:
    if not isinstance(entry, dict):
        raise ValueError("Each payload must be an object")
    if entry.get("kind") != "rule":
        raise ValueError("Each payload's kind must be 'rule'")

    rule = entry.get("rule")
    if not isinstance(rule, dict):
        raise ValueError("Each payload must carry a rule object")

    content = rule.get("content")
    if not isinstance(content, dict):
        raise ValueError("rule.content must be an object")

    condition = _parse_optional_string(content.get("condition"), field="rule.content.condition")
    description = _parse_optional_string(
        content.get("description"), field="rule.content.description"
    )
    if not condition and not description:
        raise ValueError("rule.content requires a condition, a description, or both")

    operation_value = rule.get("operation")
    operations = {operation.value: operation for operation in PayloadOperation}
    if not isinstance(operation_value, str) or operation_value not in operations:
        raise ValueError(f"rule.operation must be one of {sorted(operations)}")

    relativity_proposition = _parse_bool(
        rule.get("relativity_proposition"), field="rule.relativity_proposition"
    )
    signal_proposition = _parse_bool(
        rule.get("signal_proposition"), field="rule.signal_proposition"
    )
    title_proposition = _parse_bool(rule.get("title_proposition"), field="rule.title_proposition")

    if not (relativity_proposition or signal_proposition or title_proposition):
        raise ValueError("At least one proposition flag must be enabled per payload")

    agent_id = (
        _parse_optional_string(rule.get("agent_id"), field="rule.agent_id") or default_agent_id
    )
    updated_id = _parse_optional_string(rule.get("updated_id"), field="rule.updated_id")

    return RulePayload(
        operation=operations[operation_value],
        condition=condition,
        content=description,
        relativity_proposition=relativity_proposition,
        signal_proposition=signal_proposition,
        title_proposition=title_proposition,
        title=_parse_optional_string(rule.get("title"), field="rule.title"),
        agent_id=AgentId(agent_id) if agent_id else None,
        updated_id=RuleId(updated_id) if updated_id else None,
    )


def _serialize_indexing_task(indexing_task: IndexingTask) -> dict[str, Any]:
    return {
        "id": str(indexing_task.id),
        "status": indexing_task.status.name.lower(),
        "progress": indexing_task.progress,
        "creation_utc": indexing_task.creation_utc.isoformat(),
        "error": indexing_task.error,
        "groups": [str(group_id) for group_id in indexing_task.groups],
        "invoices": [_serialize_invoice(invoice) for invoice in indexing_task.invoices],
    }


def _serialize_invoice(invoice: Invoice) -> dict[str, Any]:
    return {
        "payload": {
            "kind": "rule",
            "rule": _serialize_rule_payload(invoice.payload),
        },
        "checksum": invoice.checksum,
        "approved": invoice.approved,
        "data": {"rule": _serialize_invoice_rule_data(invoice.data)} if invoice.data else None,
        "error": invoice.error,
    }


def _serialize_rule_payload(payload: RulePayload) -> dict[str, Any]:
    return {
        "content": {
            "condition": payload.condition,
            "content": payload.content,
        },
        "operation": payload.operation.value,
        "updated_id": str(payload.updated_id) if payload.updated_id else None,
        "title": payload.title,
        "agent_id": str(payload.agent_id) if payload.agent_id else None,
        "relativity_proposition": payload.relativity_proposition,
        "signal_proposition": payload.signal_proposition,
        "title_proposition": payload.title_proposition,
    }


def _serialize_invoice_rule_data(data: InvoiceRuleData) -> dict[str, Any]:
    return {
        "references_other_policies_proposition": data.references_other_policies_proposition,
        "signals_proposition": list(data.signals_proposition)
        if data.signals_proposition is not None
        else None,
        "anti_signals_proposition": list(data.anti_signals_proposition)
        if data.anti_signals_proposition is not None
        else None,
        "title_proposition": data.title_proposition,
    }
