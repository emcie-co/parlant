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

from pytest import fixture

from parlant.core.application import Application
from parlant.core.container import Container
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher


EXPECTED_RULE_FIELDS = {
    "id",
    "condition",
    "content",
    "title",
    "weight",
    "enabled",
    "groups",
    "metadata",
    "creation_utc",
    "modified_utc",
    "effort",
    "labels",
    "priority",
    "signals",
    "anti_signals",
    "message_review",
    "tool_review",
}


@fixture
async def dispatcher(container: Container) -> TunnelRequestDispatcher:
    return TunnelRequestDispatcher(
        application=container[Application],
        health_reporter=container[HealthReporter],
        logger=container[Logger],
    )


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


async def test_that_a_rule_can_be_created_with_condition_only(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    result = await call(
        dispatcher,
        "rules.create",
        {"condition": "the customer asks for a refund"},
    )

    rule = result["rule"]
    assert rule["condition"] == "the customer asks for a refund"
    assert rule["content"] is None
    assert rule["weight"] == "medium"
    assert rule["enabled"] is True


async def test_that_a_rule_can_be_created_with_description_only(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    result = await call(
        dispatcher,
        "rules.create",
        {"content": "always answer in French"},
    )

    rule = result["rule"]
    assert rule["content"] == "always answer in French"
    assert rule["condition"] is None


async def test_that_creating_a_rule_without_condition_or_description_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    await call_expecting_error(dispatcher, "rules.create", {"title": "Just a title"})


async def test_that_a_serialized_rule_carries_exactly_the_documented_fields(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    result = await call(
        dispatcher,
        "rules.create",
        {
            "condition": "the customer asks about shipping",
            "content": "explain shipping options",
            "title": "Shipping",
            "weight": "high",
            "effort": "low",
            "metadata": {"team": "support"},
            "labels": ["shipping"],
            "priority": 3,
            "signals": ["shipping question"],
            "anti_signals": ["billing question"],
            "message_review": True,
            "tool_review": True,
        },
    )

    rule = result["rule"]
    assert set(rule.keys()) == EXPECTED_RULE_FIELDS
    assert rule["title"] == "Shipping"
    assert rule["weight"] == "high"
    assert rule["effort"] == "low"
    assert rule["metadata"] == {"team": "support"}
    assert rule["labels"] == ["shipping"]
    assert rule["priority"] == 3
    assert rule["signals"] == ["shipping question"]
    assert rule["anti_signals"] == ["billing question"]
    assert rule["message_review"] is True
    assert rule["tool_review"] is True


async def test_that_removed_rule_concepts_are_not_serialized(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    result = await call(dispatcher, "rules.create", {"condition": "anything"})

    rule = result["rule"]
    for removed in ("action", "criticality", "composition_mode", "track"):
        assert removed not in rule


async def test_that_an_invalid_weight_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    error = await call_expecting_error(
        dispatcher,
        "rules.create",
        {"condition": "anything", "weight": "critical"},
    )
    assert "weight" in error


async def test_that_an_invalid_effort_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    error = await call_expecting_error(
        dispatcher,
        "rules.create",
        {"condition": "anything", "effort": "ultra"},
    )
    assert "effort" in error


async def test_that_a_rule_can_be_retrieved_with_its_relationships_and_tool_associations(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    first = (await call(dispatcher, "rules.create", {"condition": "first condition"}))["rule"]
    second = (await call(dispatcher, "rules.create", {"condition": "second condition"}))["rule"]

    await call(
        dispatcher,
        "relationships.create",
        {
            "source_rule": first["id"],
            "target_rule": second["id"],
            "kind": "priority",
        },
    )

    result = await call(dispatcher, "rules.retrieve", {"rule_id": first["id"]})

    assert result["rule"]["id"] == first["id"]
    assert result["tool_associations"] == []
    assert len(result["relationships"]) == 1
    assert result["relationships"][0]["kind"] == "priority"


async def test_that_a_rule_can_be_updated(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (await call(dispatcher, "rules.create", {"condition": "old condition"}))["rule"]

    result = await call(
        dispatcher,
        "rules.update",
        {
            "rule_id": created["id"],
            "condition": "new condition",
            "weight": "low",
            "metadata": {"set": {"stage": "beta"}, "unset": []},
            "labels": {"upsert": ["beta"], "remove": []},
            "message_review": True,
        },
    )

    rule = result["rule"]
    assert rule["condition"] == "new condition"
    assert rule["weight"] == "low"
    assert rule["metadata"]["stage"] == "beta"
    assert rule["labels"] == ["beta"]
    assert rule["message_review"] is True
    assert set(result.keys()) == {"rule", "relationships", "tool_associations"}


async def test_that_rules_can_be_listed_and_deleted(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (await call(dispatcher, "rules.create", {"condition": "to be deleted"}))["rule"]

    listing = await call(dispatcher, "rules.list", {})
    assert any(r["id"] == created["id"] for r in listing["rules"])

    await call(dispatcher, "rules.delete", {"rule_id": created["id"]})

    listing_after = await call(dispatcher, "rules.list", {})
    assert not any(r["id"] == created["id"] for r in listing_after["rules"])


async def test_that_a_rule_can_be_created_in_a_group(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    group = (await call(dispatcher, "groups.create", {"name": "Support Rules"}))["group"]

    created = (
        await call(
            dispatcher,
            "rules.create",
            {"condition": "grouped condition", "groups": [group["id"]]},
        )
    )["rule"]

    assert created["groups"] == [group["id"]]

    listing = await call(dispatcher, "rules.list", {"group_id": group["id"]})
    assert [r["id"] for r in listing["rules"]] == [created["id"]]
