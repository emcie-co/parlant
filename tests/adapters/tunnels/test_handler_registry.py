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

from collections.abc import AsyncIterator
from typing import Any

import pytest

from parlant.core.application import Application
from parlant.core.container import Container
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.tunnel_handlers import (
    TunnelHandlerSet,
    create_tunnel_handler_registry,
    merge_handler_sets,
)


EXPECTED_UNARY_METHODS = {
    "system.version",
    "system.health",
    # sessions
    "sessions.create",
    "sessions.read",
    "sessions.list",
    "sessions.update",
    "sessions.delete",
    "sessions.delete_many",
    "sessions.create_event",
    "sessions.list_events",
    "sessions.read_event",
    "sessions.update_event",
    "sessions.delete_events",
    # agents
    "agents.create",
    "agents.list",
    "agents.retrieve",
    "agents.update",
    "agents.delete",
    # customers
    "customers.create",
    "customers.list",
    "customers.retrieve",
    "customers.update",
    "customers.delete",
    # groups
    "groups.create",
    "groups.list",
    "groups.retrieve",
    "groups.update",
    "groups.delete",
    # rules
    "rules.create",
    "rules.list",
    "rules.retrieve",
    "rules.update",
    "rules.delete",
    # indexing
    "indexing.create",
    "indexing.retrieve",
    # relationships
    "relationships.create",
    "relationships.list",
    "relationships.retrieve",
    "relationships.delete",
    # glossary terms
    "terms.create",
    "terms.list",
    "terms.retrieve",
    "terms.update",
    "terms.delete",
    # canned responses
    "canned_responses.create",
    "canned_responses.list",
    "canned_responses.retrieve",
    "canned_responses.update",
    "canned_responses.delete",
    # context variables
    "context_variables.create",
    "context_variables.list",
    "context_variables.retrieve",
    "context_variables.resolve",
    "context_variables.update",
    "context_variables.delete",
    "context_variables.delete_many",
    "context_variables.list_values",
    "context_variables.retrieve_value",
    "context_variables.update_value",
    "context_variables.delete_value",
    # services
    "services.update",
    "services.list",
    "services.retrieve",
    "services.delete",
}

EXPECTED_STREAM_METHODS = {
    "sessions.stream_events",
    "sessions.stream_event",
}


async def test_that_the_registry_exposes_exactly_the_documented_unary_surface(
    container: Container,
) -> None:
    registry = create_tunnel_handler_registry(
        container[Application], container[HealthReporter], container[Logger]
    )

    assert set(registry.unary.keys()) == EXPECTED_UNARY_METHODS


async def test_that_the_registry_exposes_both_session_stream_methods(
    container: Container,
) -> None:
    registry = create_tunnel_handler_registry(
        container[Application], container[HealthReporter], container[Logger]
    )

    assert set(registry.streams.keys()) == EXPECTED_STREAM_METHODS


async def test_that_the_registry_does_not_expose_removed_features(
    container: Container,
) -> None:
    registry = create_tunnel_handler_registry(
        container[Application], container[HealthReporter], container[Logger]
    )

    for method in registry.unary:
        namespace = method.split(".", 1)[0]
        assert namespace not in ("journeys", "capabilities", "tags", "evaluations")


def _unary_set(*methods: str) -> TunnelHandlerSet:
    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {}

    return TunnelHandlerSet(unary={m: handler for m in methods}, streams={})


def _stream_set(*methods: str) -> TunnelHandlerSet:
    async def handler(params: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        yield {}

    return TunnelHandlerSet(unary={}, streams={m: handler for m in methods})


def test_that_merging_handler_sets_with_duplicate_unary_methods_is_rejected() -> None:
    with pytest.raises(ValueError, match="a.method"):
        merge_handler_sets(_unary_set("a.method"), _unary_set("a.method"))


def test_that_merging_handler_sets_with_duplicate_stream_methods_is_rejected() -> None:
    with pytest.raises(ValueError, match="a.stream"):
        merge_handler_sets(_stream_set("a.stream"), _stream_set("a.stream"))


def test_that_a_method_may_be_registered_as_both_unary_and_stream() -> None:
    merged = merge_handler_sets(_unary_set("a.method"), _stream_set("a.method"))

    assert "a.method" in merged.unary
    assert "a.method" in merged.streams
