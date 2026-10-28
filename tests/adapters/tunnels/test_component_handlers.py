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

"""Classical round-trip tests for the per-domain tunnel handlers.

Every test drives the real dispatcher over the container's real Application
(in-memory stores) — no module mocks. The rules and indexing handlers have
their own dedicated files.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from pytest import fixture, mark

from parlant.core.application import Application
from parlant.core.application_context import ApplicationContext
from parlant.core.container import Container
from parlant.core.health import (
    HealthReport,
    HealthReporter,
    OverallHealth,
    ReportRetention,
    StatusCriticality,
    ViewSnapshot,
)
from parlant.core.loggers import Logger
from parlant.core.nlp.model_routing import (
    ModelConfig,
    ModelConfigurationService,
    ModelScope,
    SecretRef,
    SecretValue,
)
from parlant.core.tunnel_handlers.agents import _serialize_model_routes
from parlant.core.tunnels import TunnelRequest, TunnelRequestDispatcher


def test_that_model_route_serialization_includes_reasoning_and_keeps_function_providers() -> None:
    async def provide_ranker(context: Any) -> Any:  # a ModelProvider (callable model source)
        raise NotImplementedError

    serialized = _serialize_model_routes(
        {
            "react": ModelConfig("openai:gpt-5", reasoning="high"),
            "matching.rank": ModelConfig(provide_ranker),
            "*": "openrouter",
        }
    )

    react = cast(dict[str, Any], serialized["react"])
    matching_rank = cast(dict[str, Any], serialized["matching.rank"])

    assert react["reasoning"] == {"effort": "high"}
    assert serialized["*"] == "openrouter"
    # A function-provider route is represented by its function name, not dropped.
    assert matching_rank["model"] == "provide_ranker"
    assert matching_rank["reasoning"] is None


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


###############################################################################
# Agents
###############################################################################


async def test_that_an_agent_round_trips_with_moderation_and_review_flags(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (
        await call(
            dispatcher,
            "agents.create",
            {
                "name": "Support Agent",
                "instructions": "Handles support",
                # "on" would require a configured moderation model route;
                # "off" waives the checkpoints explicitly.
                "moderation": "off",
                "message_review": True,
                "tool_review": True,
            },
        )
    )["agent"]

    assert created["moderation"] == "off"
    assert created["message_review"] is True
    assert created["tool_review"] is True
    assert isinstance(created["engine"], str)
    assert created["models"] == {}

    retrieved = (await call(dispatcher, "agents.retrieve", {"agent_id": created["id"]}))["agent"]
    assert retrieved["moderation"] == "off"


async def test_that_removed_agent_concepts_are_not_serialized_but_models_are(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (await call(dispatcher, "agents.create", {"name": "Plain Agent"}))["agent"]

    assert "composition_mode" not in created
    assert created["models"] == {}
    assert created["moderation"] == "auto"


async def test_that_agent_models_round_trip_with_recursive_secret_redaction(
    dispatcher: TunnelRequestDispatcher,
    container: Container,
) -> None:
    created = (await call(dispatcher, "agents.create", {"name": "Configured Agent"}))["agent"]
    await container[ModelConfigurationService].configure(
        ModelScope.agent(created["id"]),
        {
            "review.tools": ModelConfig(
                "openai:gpt-4o",
                args={
                    "plain": "visible",
                    "nullable": None,
                    "auth": {
                        "reference": SecretRef("production/openai/api-key"),
                        "inline": SecretValue("must-never-leave-compass"),
                    },
                    "fallbacks": [SecretRef("backup/key"), None, 2],
                },
            ),
            "matching.rank": ModelConfig("openai:gpt-4o-mini", reasoning="minimal"),
            "*": "openai",
        },
    )

    expected_models = {
        "review.tools": {
            "model": "openai:gpt-4o",
            "args": {
                "plain": "visible",
                "nullable": None,
                "auth": {
                    "reference": {"$secret": "[REDACTED]"},
                    "inline": "[REDACTED]",
                },
                "fallbacks": [{"$secret": "[REDACTED]"}, None, 2],
            },
            "rate_limit": None,
            "reasoning": None,
        },
        "matching.rank": {
            "model": "openai:gpt-4o-mini",
            "args": {},
            "rate_limit": None,
            "reasoning": {"effort": "minimal"},
        },
        "*": "openai",
    }

    retrieved = (await call(dispatcher, "agents.retrieve", {"agent_id": created["id"]}))["agent"]
    listing = await call(dispatcher, "agents.list", {})
    listed = next(agent for agent in listing["agents"] if agent["id"] == created["id"])
    updated = (
        await call(
            dispatcher,
            "agents.update",
            {"agent_id": created["id"], "instructions": "Updated"},
        )
    )["agent"]

    assert retrieved["models"] == expected_models
    assert listed["models"] == expected_models
    assert updated["models"] == expected_models
    assert "production/openai/api-key" not in repr(retrieved)
    assert "must-never-leave-compass" not in repr(retrieved)


async def test_that_an_agent_can_be_updated_and_deleted(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (await call(dispatcher, "agents.create", {"name": "Old Name"}))["agent"]

    updated = (
        await call(
            dispatcher,
            "agents.update",
            {"agent_id": created["id"], "name": "New Name", "moderation": "off"},
        )
    )["agent"]
    assert updated["name"] == "New Name"
    assert updated["moderation"] == "off"

    await call(dispatcher, "agents.delete", {"agent_id": created["id"]})
    listing = await call(dispatcher, "agents.list", {})
    assert not any(a["id"] == created["id"] for a in listing["agents"])


async def test_that_an_invalid_moderation_value_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    error = await call_expecting_error(
        dispatcher,
        "agents.create",
        {"name": "Bad Agent", "moderation": "paranoid"},
    )
    assert "moderation" in error


###############################################################################
# Sessions
###############################################################################


async def _create_session(dispatcher: TunnelRequestDispatcher) -> dict[str, Any]:
    agent = (await call(dispatcher, "agents.create", {"name": "Session Agent"}))["agent"]
    customer = (await call(dispatcher, "customers.create", {"name": "Session Customer"}))[
        "customer"
    ]

    # The session wire shape is envelope-less and keys the id as
    # "session_id" — preserved from the pre-split tunnel contract.
    return dict(
        await call(
            dispatcher,
            "sessions.create",
            {"agent_id": agent["id"], "customer_id": customer["id"]},
        )
    )


async def test_that_a_session_event_round_trips_without_a_moderation_param(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    session = await _create_session(dispatcher)

    created = await call(
        dispatcher,
        "sessions.create_event",
        {
            "session_id": session["session_id"],
            "kind": "message",
            "source": "customer",
            "message": "Hello there",
            "trigger_processing": False,
            # Legacy param from the per-message moderation era; deliberately
            # tolerated and ignored (moderation is an agent property now).
            "moderation": "none",
        },
    )

    events = await call(
        dispatcher,
        "sessions.list_events",
        {"session_id": session["session_id"], "wait_for_data": 0},
    )
    assert any(e["id"] == created["event_id"] for e in events["events"])


async def test_that_a_tunnel_message_without_a_participant_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    # A non-customer message routes through the raw create_event path; without a
    # participant it must be rejected, not silently stored (which used to corrupt the
    # session — the engine reads participant on every turn).
    session = await _create_session(dispatcher)

    error = await call_expecting_error(
        dispatcher,
        "sessions.create_event",
        {
            "session_id": session["session_id"],
            "kind": "message",
            "source": "ai_agent",
            "data": {"message": "no participant here"},
        },
    )

    assert "participant" in error


async def test_that_a_tunnel_on_behalf_message_is_stamped_with_the_agent_participant(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    # The agent participant is server-known; the tunnel constructs it rather than
    # trusting the caller's data.
    session = await _create_session(dispatcher)

    await call(
        dispatcher,
        "sessions.create_event",
        {
            "session_id": session["session_id"],
            "kind": "message",
            "source": "human_agent_on_behalf_of_ai_agent",
            "message": "Replying on behalf of the agent",
        },
    )

    events = await call(
        dispatcher,
        "sessions.list_events",
        {"session_id": session["session_id"], "wait_for_data": 0},
    )
    message = next(e for e in events["events"] if e["kind"] == "message")
    assert message["data"]["participant"]["display_name"] == "Session Agent"


async def test_that_session_metadata_updates_guard_the_reserved_namespace(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    session = await _create_session(dispatcher)

    # Session metadata updates take a FLAT public map with replace-all
    # semantics — preserved from the pre-split tunnel contract.
    updated = await call(
        dispatcher,
        "sessions.update",
        {
            "session_id": session["session_id"],
            "metadata": {"public_key": "value"},
        },
    )
    assert updated["metadata"]["public_key"] == "value"

    error = await call_expecting_error(
        dispatcher,
        "sessions.update",
        {
            "session_id": session["session_id"],
            "metadata": {"__parlant__": {"sneaky": True}},
        },
    )
    assert "reserved" in error.lower() or "__parlant__" in error


async def test_that_event_metadata_updates_guard_the_reserved_namespace(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    session = await _create_session(dispatcher)

    created = await call(
        dispatcher,
        "sessions.create_event",
        {
            "session_id": session["session_id"],
            "kind": "message",
            "source": "customer",
            "message": "Annotate me",
            "trigger_processing": False,
        },
    )

    updated = await call(
        dispatcher,
        "sessions.update_event",
        {
            "session_id": session["session_id"],
            "event_id": created["event_id"],
            "metadata": {"note": "public"},
        },
    )
    assert updated["metadata"]["note"] == "public"

    error = await call_expecting_error(
        dispatcher,
        "sessions.update_event",
        {
            "session_id": session["session_id"],
            "event_id": created["event_id"],
            "metadata": {"__parlant__": {"sneaky": True}},
        },
    )
    assert "reserved" in error.lower() or "__parlant__" in error


###############################################################################
# Customers
###############################################################################


async def test_that_a_customer_round_trips_and_incremental_listing_filters(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (
        await call(
            dispatcher,
            "customers.create",
            {"name": "Ingested Customer", "metadata": {"tier": "gold"}},
        )
    )["customer"]

    assert created["metadata"] == {"tier": "gold"}
    assert "modified_utc" in created

    unfiltered = await call(dispatcher, "customers.list", {})
    assert any(c["id"] == created["id"] for c in unfiltered["customers"])

    filtered = await call(
        dispatcher,
        "customers.list",
        {"min_modified_utc": "2999-01-01T00:00:00+00:00"},
    )
    assert not any(c["id"] == created["id"] for c in filtered["customers"])

    updated = (
        await call(
            dispatcher,
            "customers.update",
            {"customer_id": created["id"], "name": "Renamed Customer"},
        )
    )["customer"]
    assert updated["name"] == "Renamed Customer"

    await call(dispatcher, "customers.delete", {"customer_id": created["id"]})


async def test_that_customers_can_be_searched_by_name_prefix_through_the_tunnel(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    jane = (await call(dispatcher, "customers.create", {"name": "Jane Doe"}))["customer"]
    (await call(dispatcher, "customers.create", {"name": "Bob Stone"}))["customer"]

    result = await call(dispatcher, "customers.list", {"name_prefix": "jan"})

    assert [c["id"] for c in result["customers"]] == [jane["id"]]


async def test_that_customer_kind_survives_create_retrieve_update(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (
        await call(dispatcher, "customers.create", {"name": "Test persona", "kind": "persona"})
    )["customer"]
    assert created["kind"] == "persona"
    retrieved = (await call(dispatcher, "customers.retrieve", {"customer_id": created["id"]}))[
        "customer"
    ]
    assert retrieved["kind"] == "persona"
    updated = (
        await call(
            dispatcher,
            "customers.update",
            {"customer_id": created["id"], "name": "Renamed persona", "kind": "real"},
        )
    )["customer"]
    assert updated["kind"] == "persona"
    real = (await call(dispatcher, "customers.create", {"name": "Real customer"}))["customer"]
    assert real["kind"] == "real"


async def test_that_customer_kind_filter_separates_personas_and_real_customers(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    persona = (await call(dispatcher, "customers.create", {"name": "Persona", "kind": "persona"}))[
        "customer"
    ]
    real = (await call(dispatcher, "customers.create", {"name": "Real customer"}))["customer"]
    personas = await call(dispatcher, "customers.list", {"kind": "persona"})
    assert [customer["id"] for customer in personas["customers"]] == [persona["id"]]
    assert personas["total_count"] == 1
    assert personas["has_more"] is False
    reals = await call(dispatcher, "customers.list", {"kind": "real"})
    assert {customer["id"] for customer in reals["customers"]} == {real["id"], "guest"}
    assert all(customer["kind"] == "real" for customer in reals["customers"])
    assert reals["total_count"] == 2


async def test_that_persona_filter_survives_incremental_paging(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    expected_ids = set()
    for index in range(3):
        await call(dispatcher, "customers.create", {"name": f"Real {index}"})
        persona = (
            await call(
                dispatcher, "customers.create", {"name": f"Persona {index}", "kind": "persona"}
            )
        )["customer"]
        expected_ids.add(persona["id"])
    params: dict[str, Any] = {
        "kind": "persona",
        "limit": 1,
        "min_modified_utc": "1970-01-01T00:00:00+00:00",
        "sort_direction": "asc",
    }
    seen: list[str] = []
    for _ in range(4):
        page = await call(dispatcher, "customers.list", params)
        assert page["total_count"] == 3
        assert all(customer["kind"] == "persona" for customer in page["customers"])
        seen.extend(customer["id"] for customer in page["customers"])
        if not page["has_more"]:
            assert page["next_cursor"] is None
            break
        assert page["next_cursor"]
        params["cursor"] = page["next_cursor"]
    assert len(seen) == 3
    assert set(seen) == expected_ids


async def test_that_invalid_customer_kind_is_rejected(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    for kind in ("test", "PERSONA", "", True, 42, {"kind": "persona"}):
        for method, params in (
            ("customers.create", {"name": "Invalid", "kind": kind}),
            ("customers.list", {"kind": kind}),
        ):
            error = await call_expecting_error(dispatcher, method, params)
            assert "kind" in error.lower()


###############################################################################
# Groups
###############################################################################


async def test_that_a_group_round_trips(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (await call(dispatcher, "groups.create", {"name": "VIP"}))["group"]
    assert set(created.keys()) == {"id", "name"}

    retrieved = (await call(dispatcher, "groups.retrieve", {"group_id": created["id"]}))["group"]
    assert retrieved["name"] == "VIP"

    updated = (
        await call(
            dispatcher,
            "groups.update",
            {"group_id": created["id"], "name": "VIP Customers"},
        )
    )["group"]
    assert updated["name"] == "VIP Customers"

    listing = await call(dispatcher, "groups.list", {})
    assert any(g["id"] == created["id"] for g in listing["groups"])

    await call(dispatcher, "groups.delete", {"group_id": created["id"]})


async def test_that_the_tags_namespace_is_gone(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    for method in ("tags.list", "tags.retrieve"):
        response = await dispatcher.dispatch(
            TunnelRequest(request_id="req-1", method=method, params={})
        )
        assert isinstance(response.error, str)
        assert "unknown" in response.error.lower() or "method" in response.error.lower()


###############################################################################
# Glossary terms
###############################################################################


async def test_that_a_term_round_trips(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (
        await call(
            dispatcher,
            "terms.create",
            {
                "name": "Netting",
                "description": "Offsetting the value of multiple positions",
                "synonyms": ["offsetting"],
            },
        )
    )["term"]

    assert created["name"] == "Netting"
    assert created["synonyms"] == ["offsetting"]

    updated = (
        await call(
            dispatcher,
            "terms.update",
            {"term_id": created["id"], "description": "Updated description"},
        )
    )["term"]
    assert updated["description"] == "Updated description"

    listing = await call(dispatcher, "terms.list", {})
    assert any(t["id"] == created["id"] for t in listing["terms"])

    await call(dispatcher, "terms.delete", {"term_id": created["id"]})


###############################################################################
# Canned responses
###############################################################################


async def test_that_a_canned_response_round_trips(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (
        await call(
            dispatcher,
            "canned_responses.create",
            {
                "value": "Thanks for reaching out, {{customer_name}}!",
                "note": "Use a warm tone.",
            },
        )
    )["canned_response"]

    assert created["value"] == "Thanks for reaching out, {{customer_name}}!"
    assert created["note"] == "Use a warm tone."

    updated = (
        await call(
            dispatcher,
            "canned_responses.update",
            {
                "canned_response_id": created["id"],
                "value": "Thanks!",
                "note": "Keep it concise.",
            },
        )
    )["canned_response"]
    assert updated["value"] == "Thanks!"
    assert updated["note"] == "Keep it concise."

    listing = await call(dispatcher, "canned_responses.list", {})
    assert any(c["id"] == created["id"] for c in listing["canned_responses"])

    await call(dispatcher, "canned_responses.delete", {"canned_response_id": created["id"]})


###############################################################################
# Context variables
###############################################################################


async def test_that_resolve_reports_which_source_won_for_each_variable(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    # The Customers drawer badges every variable with the source the engine
    # would actually use: customer beats group beats agent beats default.
    # This resolves through the same precedence the engine's VariableLoader
    # walks, read-only, so the badge cannot drift from the engine's truth.
    agent = (await call(dispatcher, "agents.create", {"name": "Drawer Agent"}))["agent"]
    group = (await call(dispatcher, "groups.create", {"name": "enterprise"}))["group"]
    customer = (
        await call(
            dispatcher,
            "customers.create",
            {"name": "Reviewed Customer", "groups": [group["id"]]},
        )
    )["customer"]

    plan = (await call(dispatcher, "context_variables.create", {"name": "plan"}))[
        "context_variable"
    ]
    language = (await call(dispatcher, "context_variables.create", {"name": "language"}))[
        "context_variable"
    ]
    theme = (await call(dispatcher, "context_variables.create", {"name": "theme"}))[
        "context_variable"
    ]

    # plan: set for the customer AND the group -- the customer's own value wins.
    await call(
        dispatcher,
        "context_variables.update_value",
        {"variable_id": plan["id"], "key": customer["id"], "data": {"tier": "pro"}},
    )
    await call(
        dispatcher,
        "context_variables.update_value",
        {"variable_id": plan["id"], "key": f"group:{group['id']}", "data": {"tier": "team"}},
    )
    # language: group value AND the global default -- the group wins.
    await call(
        dispatcher,
        "context_variables.update_value",
        {"variable_id": language["id"], "key": f"group:{group['id']}", "data": "es-MX"},
    )
    await call(
        dispatcher,
        "context_variables.update_value",
        {"variable_id": language["id"], "key": "DEFAULT", "data": "en-US"},
    )
    # theme: only the global default exists.
    await call(
        dispatcher,
        "context_variables.update_value",
        {"variable_id": theme["id"], "key": "DEFAULT", "data": "light"},
    )

    resolved = (
        await call(
            dispatcher,
            "context_variables.resolve",
            {"customer_id": customer["id"], "agent_id": agent["id"]},
        )
    )["resolved"]

    by_name = {entry["context_variable"]["name"]: entry for entry in resolved}

    assert by_name["plan"]["value"]["data"] == {"tier": "pro"}
    assert by_name["plan"]["source"] == {"kind": "customer"}

    assert by_name["language"]["value"]["data"] == "es-MX"
    assert by_name["language"]["source"] == {
        "kind": "group",
        "group_id": group["id"],
        "group_name": "enterprise",
    }

    assert by_name["theme"]["value"]["data"] == "light"
    assert by_name["theme"]["source"] == {"kind": "default"}

    # Every value carries its own modified time -- the drawer shows it.
    assert all(entry["value"]["modified_utc"] for entry in resolved)

    # A variable with no stored value anywhere is not in the answer at all:
    # the drawer shows what the engine would load, and the engine loads
    # nothing for it.
    unset = (await call(dispatcher, "context_variables.create", {"name": "unset"}))[
        "context_variable"
    ]
    resolved_again = (
        await call(
            dispatcher,
            "context_variables.resolve",
            {"customer_id": customer["id"], "agent_id": agent["id"]},
        )
    )["resolved"]
    assert unset["id"] not in {e["context_variable"]["id"] for e in resolved_again}


async def test_that_a_context_variable_and_its_values_round_trip(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    created = (
        await call(
            dispatcher,
            "context_variables.create",
            {"name": "Plan Tier", "description": "The customer's plan"},
        )
    )["context_variable"]

    updated = (
        await call(
            dispatcher,
            "context_variables.update",
            {"variable_id": created["id"], "name": "Plan"},
        )
    )["context_variable"]
    assert updated["name"] == "Plan"

    await call(
        dispatcher,
        "context_variables.update_value",
        {"variable_id": created["id"], "key": "customer-1", "data": {"tier": "pro"}},
    )

    value = (
        await call(
            dispatcher,
            "context_variables.retrieve_value",
            {"variable_id": created["id"], "key": "customer-1"},
        )
    )["value"]
    assert value["data"] == {"tier": "pro"}

    values = await call(
        dispatcher,
        "context_variables.list_values",
        {"variable_id": created["id"]},
    )
    assert any(v["key"] == "customer-1" for v in values["values"])

    await call(
        dispatcher,
        "context_variables.delete_value",
        {"variable_id": created["id"], "key": "customer-1"},
    )

    listing = await call(dispatcher, "context_variables.list", {})
    assert any(v["id"] == created["id"] for v in listing["context_variables"])

    await call(dispatcher, "context_variables.delete", {"variable_id": created["id"]})

    listing_after = await call(dispatcher, "context_variables.list", {})
    assert not any(v["id"] == created["id"] for v in listing_after["context_variables"])


async def test_that_context_variables_can_be_deleted_in_bulk(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    for name in ("Bulk A", "Bulk B"):
        await call(dispatcher, "context_variables.create", {"name": name, "description": ""})

    await call(dispatcher, "context_variables.delete_many", {})

    listing = await call(dispatcher, "context_variables.list", {})
    assert listing["context_variables"] == []


###############################################################################
# Services
###############################################################################


@mark.parametrize("narration", ["Looking it up", ["Checking", "Reading"], None])
async def test_that_tool_narration_survives_service_tunnel_reads(
    dispatcher: TunnelRequestDispatcher,
    narration: str | list[str] | None,
) -> None:
    from parlant.core.services.tools.plugins import tool
    from parlant.core.tools import ToolContext, ToolResult
    from tests.test_utilities import run_service_server

    decorator = tool(narration=narration) if narration is not None else tool()

    @decorator
    def lookup(context: ToolContext) -> ToolResult:
        """Read a diagnostic value."""
        return ToolResult({"found": True})

    async with run_service_server([lookup]) as server:
        await call(
            dispatcher,
            "services.update",
            {"name": "narration-test", "kind": "sdk", "sdk": {"url": server.url}},
        )
        listing = await call(dispatcher, "services.list", {})
        retrieved = await call(dispatcher, "services.retrieve", {"name": "narration-test"})
        listed = next(
            service for service in listing["services"] if service["name"] == "narration-test"
        )
        for service in [listed, retrieved["service"]]:
            assert "narration" in service["tools"][0]
            assert service["tools"][0]["narration"] == narration


async def test_that_the_service_surface_handles_an_empty_registry(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    listing = await call(dispatcher, "services.list", {})
    assert listing["services"] == []

    # An unknown service resolves to null rather than an error, matching the
    # other retrieve handlers; deletion of an unknown service is an error.
    missing = await call(dispatcher, "services.retrieve", {"name": "no-such-service"})
    assert missing["service"] is None

    await call_expecting_error(dispatcher, "services.delete", {"name": "no-such-service"})


###############################################################################
# System
###############################################################################


async def test_that_system_version_reports_the_parlant_version(
    dispatcher: TunnelRequestDispatcher,
) -> None:
    from parlant.core.version import VERSION

    result = await call(dispatcher, "system.version", {})
    assert result["version"] == VERSION


@dataclass
class _FixedStatusView:
    """Minimal HealthView that always renders a chosen status."""

    name: str
    kinds: tuple[str, ...]
    criticality: StatusCriticality = StatusCriticality.CRITICAL
    fixed_status: OverallHealth = OverallHealth.HEALTHY

    def render(
        self,
        reports_by_kind: Mapping[str, Sequence[HealthReport]],
    ) -> ViewSnapshot:
        return ViewSnapshot(
            status=self.fixed_status,
            body={"sample_count": sum(len(v) for v in reports_by_kind.values())},
        )


async def test_that_system_health_reports_the_reporter_snapshot(
    dispatcher: TunnelRequestDispatcher,
    container: Container,
) -> None:
    result = await call(dispatcher, "system.health", {})

    expected = container[HealthReporter].snapshot()
    assert result["instance_id"] == expected["instance_id"]
    assert result["status"] == expected["status"]
    assert set(result["checks"]) == set(expected["checks"])


async def test_that_system_health_reports_the_worst_critical_view_status(
    container: Container,
) -> None:
    reporter = HealthReporter(ApplicationContext(instance_id="test-instance"))
    reporter.configure_retention(
        "probe",
        ReportRetention(window=timedelta(seconds=60), max_count=100),
    )
    reporter.register_view(
        _FixedStatusView(
            name="probe",
            kinds=("probe",),
            fixed_status=OverallHealth.DEGRADED,
        )
    )
    reporter.report("probe", {})

    dispatcher = TunnelRequestDispatcher(
        application=container[Application],
        logger=container[Logger],
        health_reporter=reporter,
    )

    result = await call(dispatcher, "system.health", {})

    assert result["status"] == "degraded"
    assert result["checks"]["probe"]["status"] == "degraded"
