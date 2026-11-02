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

import pytest

from parlant.core.agents import AgentId
from parlant.core.engines.compass.prompt_builder import (
    VETO,
    PromptBuilder,
    SectionAdaptation,
)
from parlant.core.nlp.model_routing import InvalidModelRouteError, ModelScope
from parlant.core.nlp.prompt_routing import PromptConfigurationService


class _CustomBuilder(PromptBuilder):
    pass


class _OtherBuilder(PromptBuilder):
    pass


async def test_that_an_unconfigured_route_yields_a_stock_prompt_builder() -> None:
    service = PromptConfigurationService()

    builder = service.create_builder("matching.rank")

    assert type(builder) is PromptBuilder


async def test_that_a_server_scoped_provider_supplies_the_builder() -> None:
    # A provider is any zero-argument callable returning a PromptBuilder —
    # a subclass itself qualifies.
    service = PromptConfigurationService()
    await service.configure(ModelScope.server(), {"matching.rank": _CustomBuilder})

    assert type(service.create_builder("matching.rank")) is _CustomBuilder
    assert type(service.create_builder("compaction")) is PromptBuilder


async def test_that_an_agent_scoped_provider_overrides_the_server_scope() -> None:
    service = PromptConfigurationService()
    await service.configure(ModelScope.server(), {"matching.rank": _CustomBuilder})
    await service.configure(ModelScope.agent("agent-1"), {"matching.rank": _OtherBuilder})

    assert (
        type(service.create_builder("matching.rank", agent_id=AgentId("agent-1"))) is _OtherBuilder
    )
    assert (
        type(service.create_builder("matching.rank", agent_id=AgentId("agent-2"))) is _CustomBuilder
    )
    assert type(service.create_builder("matching.rank")) is _CustomBuilder


async def test_that_route_wildcards_resolve_by_specificity() -> None:
    class _Global(PromptBuilder):
        pass

    class _Namespace(PromptBuilder):
        pass

    service = PromptConfigurationService()
    await service.configure(
        ModelScope.server(),
        {
            "*": _Global,
            "matching.*": _Namespace,
            "matching.rank": _CustomBuilder,
        },
    )

    assert type(service.create_builder("matching.rank")) is _CustomBuilder
    assert type(service.create_builder("matching.distill")) is _Namespace
    assert type(service.create_builder("compaction")) is _Global


async def test_that_effort_qualified_routes_win_for_their_effort() -> None:
    service = PromptConfigurationService()
    await service.configure(
        ModelScope.server(),
        {
            "matching.rank": _CustomBuilder,
            "matching.rank[effort=high]": _OtherBuilder,
        },
    )

    high = service.create_builder("matching.rank", qualifiers={"effort": "high"})
    low = service.create_builder("matching.rank", qualifiers={"effort": "low"})
    unqualified = service.create_builder("matching.rank")

    assert type(high) is _OtherBuilder
    assert type(low) is _CustomBuilder
    assert type(unqualified) is _CustomBuilder


async def test_that_build_listeners_are_attached_and_compose() -> None:
    # The factory attaches the component's on_build to whatever the provider
    # returns — composing with, never clobbering, a listener the provider
    # installed itself. Listeners keep firing once per distinct built prompt.
    provider_builds: list[str] = []
    component_builds: list[str] = []

    def provider() -> PromptBuilder:
        return PromptBuilder(on_build=provider_builds.append)

    service = PromptConfigurationService()
    await service.configure(ModelScope.server(), {"compaction": provider})

    builder = service.create_builder("compaction", on_build=component_builds.append)
    builder.add_section(name="greeting", template="hello")
    builder.build()
    builder.build()

    assert len(provider_builds) == 1
    assert "hello" in provider_builds[0]
    assert component_builds == provider_builds


async def test_that_clearing_a_provider_restores_the_fallback() -> None:
    service = PromptConfigurationService()
    await service.configure(ModelScope.server(), {"matching.rank": _CustomBuilder})
    await service.configure(ModelScope.server(), {"matching.rank": None})

    assert type(service.create_builder("matching.rank")) is PromptBuilder


async def test_that_concrete_routes_are_required_for_creation() -> None:
    service = PromptConfigurationService()

    with pytest.raises(InvalidModelRouteError):
        service.create_builder("matching.*")


# --- adapt_section: the sanctioned subclass customization seam --------------------


def test_that_adapt_section_can_replace_a_sections_template() -> None:
    class _Reworded(PromptBuilder):
        def adapt_section(
            self, name: Any, template: str, props: dict[str, Any]
        ) -> str | SectionAdaptation | None:
            if name == "greeting":
                return "howdy {who}"
            return None

    builder = _Reworded()
    builder.add_section(name="greeting", template="hello {who}", props={"who": "world"})
    builder.add_section(name="farewell", template="goodbye")

    prompt = builder.build()

    assert "howdy world" in prompt
    assert "hello world" not in prompt
    assert "goodbye" in prompt  # untouched sections pass through unchanged


def test_that_adapt_section_can_veto_a_section() -> None:
    class _Vetoing(PromptBuilder):
        def adapt_section(
            self, name: Any, template: str, props: dict[str, Any]
        ) -> str | SectionAdaptation | None:
            return VETO if name == "greeting" else None

    builder = _Vetoing()
    builder.add_section(name="greeting", template="hello")
    builder.add_section(name="farewell", template="goodbye")

    assert "greeting" not in builder.sections
    prompt = builder.build()
    assert "hello" not in prompt
    assert "goodbye" in prompt


def test_that_the_default_adapt_section_keeps_sections_unchanged() -> None:
    builder = PromptBuilder()
    builder.add_section(name="greeting", template="hello {who}", props={"who": "world"})

    assert "hello world" in builder.build()


def test_that_adapt_section_receives_the_sections_props() -> None:
    # The hook sees props so replacements can be conditional on them —
    # the case that genuinely needs logic rather than static overrides.
    class _Conditional(PromptBuilder):
        def adapt_section(
            self, name: Any, template: str, props: dict[str, Any]
        ) -> str | SectionAdaptation | None:
            if props.get("tone") == "casual":
                return "hey {who}"
            return None

    builder = _Conditional()
    builder.add_section(
        name="greeting", template="hello {who}", props={"who": "world", "tone": "casual"}
    )
    builder.add_section(name="farewell", template="goodbye {who}", props={"who": "world"})

    prompt = builder.build()

    assert "hey world" in prompt
    assert "goodbye world" in prompt
