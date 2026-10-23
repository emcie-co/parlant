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

from typing import Any, Optional, Sequence

import pytest
from pytest import fixture

from parlant.core.agents import Agent, AgentId
from parlant.core.application import Application
from parlant.core.container import Container
from parlant.core.glossary import Term
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.indexing.rule_relativity_proposer import (
    RuleRelativityProposer,
    RuleRelativityProposition,
)
from parlant.core.services.indexing.rule_signal_proposer import (
    RuleSignalProposer,
    RuleSignalProposition,
)
from parlant.core.services.indexing.rule_title_proposer import (
    RuleTitleProposer,
    RuleTitleProposition,
)
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher


@fixture(autouse=True)
def deterministic_indexing_proposers(
    container: Container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def propose_relativity(
        content: Optional[str],
        condition: Optional[str],
        title: str | None = None,
        agent_id: AgentId | None = None,
        progress_report: Optional[ProgressReport] = None,
    ) -> RuleRelativityProposition:
        return RuleRelativityProposition(references_other_policies=False)

    async def propose_signals(
        content: Optional[str],
        condition: Optional[str],
        title: str,
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        agent_id: AgentId | None = None,
        progress_report: Optional[ProgressReport] = None,
    ) -> RuleSignalProposition:
        return RuleSignalProposition(
            signals=["a signal"],
            anti_signals=["an anti-signal"],
            rationale="Test fixture",
        )

    async def propose_title(
        content: Optional[str],
        condition: Optional[str],
        agent: Agent | None,
        glossary_terms: Sequence[Term],
        agent_id: AgentId | None = None,
        progress_report: Optional[ProgressReport] = None,
    ) -> RuleTitleProposition:
        return RuleTitleProposition(title="Greeting Policy", rationale="Test fixture")

    monkeypatch.setattr(
        container[RuleRelativityProposer],
        "propose_relativity",
        propose_relativity,
    )
    monkeypatch.setattr(container[RuleSignalProposer], "propose_signals", propose_signals)
    monkeypatch.setattr(container[RuleTitleProposer], "propose_title", propose_title)


@fixture
async def dispatcher(container: Container) -> TunnelRequestDispatcher:
    return TunnelRequestDispatcher(
        application=container[Application],
        health_reporter=container[HealthReporter],
        logger=container[Logger],
    )


def _rule_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "rule",
        "rule": {
            "content": {
                "condition": "the customer greets you",
                "content": "greet them back with 'Hello'",
            },
            "operation": "add",
            "relativity_proposition": True,
        },
    }
    payload["rule"].update(overrides)
    return payload


async def call(
    dispatcher: TunnelRequestDispatcher,
    method: str,
    params: dict[str, Any],
) -> Any:
    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-1", method=method, params=params)
    )
    assert response.error is None, response.error
    return response.result


async def call_expecting_error(
    dispatcher: TunnelRequestDispatcher,
    method: str,
    params: dict[str, Any],
) -> str:
    response = await dispatcher.dispatch(
        TunnelRequest(request_id="req-1", method=method, params=params)
    )
    assert response.error is not None
    # Handlers report failures as plain strings; a dict error would mean a
    # different failure shape than these tests are asserting on.
    assert isinstance(response.error, str)
    return response.error


async def test_that_an_indexing_task_can_be_created_and_retrieved_to_completion(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = await call(
        dispatcher,
        "indexing.create",
        {"payloads": [_rule_payload()]},
    )

    task = created["indexing_task"]
    assert task["status"] in ("pending", "running", "completed")

    result = await call(
        dispatcher,
        "indexing.retrieve",
        {"indexing_task_id": task["id"], "wait_for_completion": 30},
    )

    completed = result["indexing_task"]
    assert completed["status"] == "completed"
    assert len(completed["invoices"]) == 1

    invoice = completed["invoices"][0]
    assert invoice["approved"] is True
    assert invoice["data"]["rule"]["references_other_policies_proposition"] is False


async def test_that_signal_and_title_propositions_are_serialized(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = await call(
        dispatcher,
        "indexing.create",
        {
            "payloads": [
                _rule_payload(
                    relativity_proposition=False,
                    signal_proposition=True,
                    title_proposition=True,
                )
            ]
        },
    )

    result = await call(
        dispatcher,
        "indexing.retrieve",
        {"indexing_task_id": created["indexing_task"]["id"], "wait_for_completion": 30},
    )

    data = result["indexing_task"]["invoices"][0]["data"]["rule"]
    assert data["signals_proposition"] == ["a signal"]
    assert data["anti_signals_proposition"] == ["an anti-signal"]
    assert data["title_proposition"] == "Greeting Policy"


async def test_that_the_serialized_payload_mirrors_the_rest_shape(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = await call(
        dispatcher,
        "indexing.create",
        {"payloads": [_rule_payload()]},
    )

    result = await call(
        dispatcher,
        "indexing.retrieve",
        {"indexing_task_id": created["indexing_task"]["id"], "wait_for_completion": 30},
    )

    payload = result["indexing_task"]["invoices"][0]["payload"]
    assert payload["kind"] == "rule"
    assert payload["rule"]["content"]["condition"] == "the customer greets you"
    assert payload["rule"]["operation"] == "add"
    assert "action" not in payload["rule"]["content"]
    assert "tool_ids" not in payload["rule"]


async def test_that_an_indexing_task_serializes_its_groups(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = await call(
        dispatcher,
        "indexing.create",
        {"payloads": [_rule_payload()]},
    )

    assert created["indexing_task"]["groups"] == []


async def test_that_creating_an_indexing_task_without_payloads_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    await call_expecting_error(dispatcher, "indexing.create", {"payloads": []})


async def test_that_a_payload_without_any_proposition_flag_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    error = await call_expecting_error(
        dispatcher,
        "indexing.create",
        {"payloads": [_rule_payload(relativity_proposition=False)]},
    )
    assert "proposition" in error


async def test_that_retrieval_times_out_when_completion_takes_too_long(
    dispatcher: TunnelRequestDispatcher,
    container: Container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    async def stalled_relativity(*args: Any, **kwargs: Any) -> RuleRelativityProposition:
        await asyncio.sleep(30)
        return RuleRelativityProposition(references_other_policies=False)

    monkeypatch.setattr(
        container[RuleRelativityProposer],
        "propose_relativity",
        stalled_relativity,
    )

    created = await call(
        dispatcher,
        "indexing.create",
        {"payloads": [_rule_payload()]},
    )

    error = await call_expecting_error(
        dispatcher,
        "indexing.retrieve",
        {"indexing_task_id": created["indexing_task"]["id"], "wait_for_completion": 1},
    )
    assert "timed out" in error.lower()
